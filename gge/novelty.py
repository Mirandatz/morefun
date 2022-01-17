import typing

T = typing.TypeVar("T")


def get_novelty_or_none(
    generator: typing.Callable[[], T],
    novelty_checker: typing.Callable[[T], bool],
) -> T | None:
    res = generator()

    if novelty_checker(res):
        return res
    else:
        return None
