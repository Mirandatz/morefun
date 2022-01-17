import typing

T = typing.TypeVar("T")


def collect_results_from_fallible_function(
    generator: typing.Callable[[], T | None],
    num_results: int,
    max_failures: int,
) -> list[T] | None:
    """
    An impure function is considered fallible if its return value,
    which should be of type T, is sometimes `None`.

    This function runs a fallible function multiple times in a row,
    trying to collect `num_results` valid (i.e. not None) results.
    If the fallible function produces more than `max_failure` `None`s,
    then `None` is returned instead.
    """

    assert num_results > 0
    assert max_failures >= 0

    results: list[T] = []
    failures = 0

    while len(results) < num_results and failures <= max_failures:
        res = generator()
        if res is None:
            failures += 1
        else:
            results.append(res)

    if failures <= max_failures:
        return results
    else:
        return None
