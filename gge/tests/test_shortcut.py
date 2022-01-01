import itertools
import math

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


def test_downsampling_shortcut() -> None:
    src = gl.Shape(20, 20, 3)
    dst = gl.Shape(10, 10, 5)
    actual = conn.downsampling_shortcut(src, dst, "a")
    expected = gl.Conv2D("a", filter_count=5, kernel_size=1, stride=2)
    assert expected == actual


def test_connect_downsampling_shortcut_identity() -> None:
    width = 10
    height = 10
    depth = 3

    filter_count = 128
    stride = 2

    input_layer = gl.Input(gl.Shape(width, height, depth))
    src_params = gl.Conv2D(
        name="wololo",
        filter_count=filter_count,
        kernel_size=5,
        stride=stride,
    )
    src_layer = gl.ConnectedConv2D(input_layer=input_layer, params=src_params)
    dst_shape = gl.Shape(
        width=int(width / stride),
        height=int(height / stride),
        depth=filter_count,
    )
    actual = conn.connect_downsampling_shortcut(src_layer, dst_shape, "whatever")
    expected = src_layer
    assert expected == actual


def test_connect_downsampling_shortcut() -> None:
    src_layer = gl.ConnectedBatchNorm(
        input_layer=gl.Input(gl.Shape(10, 10, 3)),
        params=gl.BatchNorm(name="wololo"),
    )
    dst_shape = gl.Shape(width=2, height=2, depth=10)

    connected_shortcut = conn.connect_downsampling_shortcut(
        src_layer, dst_shape, "whatever"
    )

    assert connected_shortcut != src_layer
    assert connected_shortcut.input_layer == src_layer
    assert connected_shortcut.output_shape == dst_shape


def test_upsampling_shortcut_to_nonlarger_dst() -> None:
    src = gl.Shape(2, 2, 2)

    for values in itertools.product([1, 2], [1, 2], [1, 2]):
        dst = gl.Shape(*values)
        with pytest.raises(AssertionError):
            _ = conn.upsampling_shortcut(src, dst, "a")


def test_upsampling_shortcut_to_incompatible_ratio() -> None:
    src = gl.Shape(10, 10, 2)
    dst = gl.Shape(20, 19, 18)

    with pytest.raises(ValueError):
        _ = conn.upsampling_shortcut(src, dst, "a")


def test_upsampling_shortcut() -> None:
    src = gl.Shape(2, 2, 2)
    dst = gl.Shape(8, 8, 5)
    actual = conn.upsampling_shortcut(src, dst, "ew")
    expected = gl.Conv2DTranspose("ew", filter_count=5, kernel_size=1, stride=4)
    assert expected == actual


def test_connect_upsampling_shortcut_identity() -> None:
    width = 10
    height = 10
    depth = 3

    filter_count = 128
    stride = 2

    input_layer = gl.Input(gl.Shape(width, height, depth))
    src_params = gl.Conv2DTranspose(
        name="wololo",
        filter_count=filter_count,
        kernel_size=5,
        stride=stride,
    )
    src_layer = gl.ConnectedConv2DTranspose(input_layer=input_layer, params=src_params)
    dst_shape = gl.Shape(
        width=width * stride,
        height=height * stride,
        depth=filter_count,
    )
    actual = conn.connect_upsampling_shortcut(src_layer, dst_shape, "whatever")
    expected = src_layer
    assert expected == actual


def test_connect_upsampling_shortcut() -> None:
    src_layer = gl.ConnectedBatchNorm(
        input_layer=gl.Input(gl.Shape(1, 2, 3)),
        params=gl.BatchNorm(name="wololo"),
    )
    dst_shape = gl.Shape(width=5, height=10, depth=19)

    connected_shortcut = conn.connect_upsampling_shortcut(
        src_layer, dst_shape, "whatever"
    )

    assert connected_shortcut != src_layer
    assert connected_shortcut.input_layer == src_layer
    assert connected_shortcut.output_shape == dst_shape
