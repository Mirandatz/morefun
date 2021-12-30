import pytest

import gge.type_checking as tc


def test_empty_tuple() -> None:
    tc.assert_tuple_is_homogeneous(tuple())


def test_empty_list() -> None:
    with pytest.raises(AssertionError):
        tc.assert_tuple_is_homogeneous([])  # type: ignore


def test_homogenous_tuple() -> None:
    tc.assert_tuple_is_homogeneous((1, 2, 3, 4))


def test_non_homogenous_tuple() -> None:
    with pytest.raises(AssertionError):
        tc.assert_tuple_is_homogeneous((1, 2, 3, "a"))
