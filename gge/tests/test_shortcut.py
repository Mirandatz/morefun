import itertools

import pytest

import gge.connections as conn
import gge.layers as gl


@pytest.fixture
def input_layer() -> gl.Input:
    return gl.Input(gl.Shape(width=2 ** 10, height=2 ** 10, depth=3))


def test_downsampling_to_non_smaller_dst() -> None:
    src = gl.Shape(1, 1, 1)

    for values in itertools.product([1, 2], [1, 2], [1, 2]):
        dst = gl.Shape(*values)
        with pytest.raises(AssertionError):
            _ = conn.downsampling_shortcut(src, dst, "a")


def test_downsampling_incompatible_ratio() -> None:
    src = gl.Shape(20, 20, 10)
    dst = gl.Shape(10, 9, 10)

    with pytest.raises(ValueError):
        _ = conn.downsampling_shortcut(src, dst, "a")


def test_connect_downsampling_shortcut() -> None:
    src = gl.Shape(20, 20, 3)
    dst = gl.Shape(10, 10, 5)
    actual = conn.downsampling_shortcut(src, dst, "a")
    expected = gl.Conv2D("a", filter_count=5, kernel_size=1, stride=2)
    assert expected == actual


def test_connected_downsampling_shortcut_identity() -> None:
    w = 10
    h = 10
    d = 3

    filter_count = 128
    stride = 2

    input_layer = gl.Input(gl.Shape(w, h, d))
    src_params = gl.Conv2D(
        name="wololo",
        filter_count=filter_count,
        kernel_size=5,
        stride=stride,
    )
    src_layer = gl.ConnectedConv2D(input_layer=input_layer, params=src_params)
    dst_shape = gl.Shape(
        width=int(w / stride),
        height=int(h / stride),
        depth=filter_count,
    )
    actual = conn.connect_downsampling_shortcut(src_layer, dst_shape, "whatever")
    expected = src_layer
    assert expected == actual
