import functools
import typing

import attrs
import hypothesis.strategies as hs
from hypothesis import given

import morefun.fallible as gf

T = typing.TypeVar("T")
DrawStrat = typing.Callable[..., typing.Any]


def identity(x: T) -> T:
    return x


@attrs.frozen
class FallibleResultsSequence:
    data: list[str | None]
    num_results: int
    max_failures: int


@hs.composite
def should_succeed_sequence_of_results(draw: DrawStrat) -> FallibleResultsSequence:
    results = draw(hs.lists(hs.just("yeey") | hs.none(), max_size=9))
    guaranteed_success = draw(hs.permutations(results + ["yeey"]))

    num_results = draw(
        hs.integers(min_value=1, max_value=guaranteed_success.count("yeey"))
    )
    max_failures = draw(hs.integers(min_value=guaranteed_success.count(None)))

    return FallibleResultsSequence(
        guaranteed_success, num_results=num_results, max_failures=max_failures
    )


@hs.composite
def should_fail_sequence_of_results(draw: DrawStrat) -> FallibleResultsSequence:
    results = draw(hs.lists(hs.just("yeey") | hs.none(), max_size=9))
    guaranteed_failure = draw(hs.permutations(results + [None]))

    num_results = draw(hs.integers(min_value=guaranteed_failure.count("yeey") + 1))
    max_failures = draw(
        hs.integers(min_value=0, max_value=guaranteed_failure.count(None) - 1)
    )

    return FallibleResultsSequence(
        guaranteed_failure, num_results=num_results, max_failures=max_failures
    )


@given(should_succeed_sequence_of_results())
def test_collect_from_function_that_should_succeed_enough(
    seq: FallibleResultsSequence,
) -> None:
    """Collecting results from a function that returns enough results should succeed."""

    data_iter = iter(seq.data)
    generator = functools.partial(next, data_iter)

    results = gf.collect_results_from_fallible_function(
        generator,
        num_results=seq.num_results,
        max_failures=seq.max_failures,
    )

    assert results is not None
    assert len(results) == seq.num_results
    assert all(b is not None for b in results)


@given(should_fail_sequence_of_results())
def test_collect_from_function_that_should_not_succeed_enough(
    seq: FallibleResultsSequence,
) -> None:
    """Collecting results from a function that does not return enough results should fail."""
    data_iter = iter(seq.data)
    generator = functools.partial(next, data_iter)

    results = gf.collect_results_from_fallible_function(
        generator,
        num_results=seq.num_results,
        max_failures=seq.max_failures,
    )

    assert results is None


@given(
    num_results=hs.integers(min_value=1, max_value=99),
    max_failures=hs.integers(min_value=0, max_value=99),
)
def test_collect_from_never_fail_function(
    num_results: int,
    max_failures: int,
) -> None:
    """Collecting results of unfallible functions should always succeed."""

    unfallible = functools.partial(identity, "hehe")

    results = gf.collect_results_from_fallible_function(
        generator=unfallible,
        num_results=num_results,
        max_failures=max_failures,
    )

    assert results is not None
    assert num_results == len(results)
    assert all(x == "hehe" for x in results)


@given(
    num_results=hs.integers(min_value=1, max_value=99),
    max_failures=hs.integers(min_value=0, max_value=99),
)
def test_collect_from_always_fail_function(
    num_results: int,
    max_failures: int,
) -> None:
    """Collecting results of guaranteed-failure functions should never succeed."""

    always_fail = functools.partial(identity, None)

    results = gf.collect_results_from_fallible_function(
        generator=always_fail,
        num_results=num_results,
        max_failures=max_failures,
    )

    assert results is None
