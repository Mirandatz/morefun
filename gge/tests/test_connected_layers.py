import dataclasses as dt

import pytest

import gge.layers as gl

INPUT_WIDTH = 2 ** 10
INPUT_HEIGHT = 2 ** 6
INPUT_DEPTH = 3
INPUT_SHAPE = gl.Shape(
    width=INPUT_WIDTH,
    height=INPUT_HEIGHT ** 10,
    depth=INPUT_DEPTH,
)
INPUT_LAYER = gl.Input(INPUT_SHAPE)


def test_conv2d_output_depth() -> None:
    conv_params = gl.Conv2D(name="a", filter_count=17, kernel_size=5, stride=4)
    connected = gl.ConnectedConv2D(input_layer=INPUT_LAYER, params=conv_params)
    actual = connected.output_shape

    expected = dt.replace(INPUT_SHAPE, width=INPUT_WIDTH / 4, height=INPUT_DEPTH / 4)
    assert expected == actual


def test_conv2d_output_depth() -> None:
    base = gl.Conv2D(name="a", filter_count=17, kernel_size=5, stride=1)
    connected = gl.ConnectedConv2D(input_layer=INPUT_LAYER, params=base)
    actual = connected.output_shape

    expected = dt.replace(INPUT_SHAPE, depth=17)
    assert expected == actual
