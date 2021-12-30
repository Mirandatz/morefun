import typing

_T = typing.TypeVar("_T")


def assert_tuple_is_homogeneous(tup: tuple[_T, ...], expected_type: type) -> None:
    assert isinstance(tup, tuple)
    if len(tup) == 0:
        return

    for item in tup:
        assert isinstance(item, expected_type)
