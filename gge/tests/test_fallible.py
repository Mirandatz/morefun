import functools
import typing

import hypothesis.strategies as hs
from hypothesis import given

import gge.fallible as gf

T = typing.TypeVar("T")


def identity(x: T) -> T:
    return x


@given(
    num_results=hs.integers(min_value=1, max_value=99),
    max_failures=hs.integers(min_value=0, max_value=99),
)
def test_collect_from_never_fail_function(
    num_results: int,
    max_failures: int,
) -> None:
    """Collecting results of unfallible functions should always succeed"""

    unfallible = functools.partial(identity, "hehe")

    results = gf.collect_results_from_fallible_function(
        func=unfallible,
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
    """Collecting results of guaranteed-failure functions should never succeed"""

    always_fail = functools.partial(identity, None)

    results = gf.collect_results_from_fallible_function(
        func=always_fail,
        num_results=num_results,
        max_failures=max_failures,
    )

    assert results is None
