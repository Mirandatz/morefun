import typing

_T = typing.TypeVar("_T")


def assert_tuple_is_homogeneous(tup: tuple[_T, ...]) -> None:
    assert isinstance(tup, tuple)
    if len(tup) == 0:
        return

    expected_type = type(tup[0])
    for item in tup[1:]:
        assert isinstance(item, expected_type)
