import itertools

import gge.layers as gl


def test_same_aspect_ratio() -> None:
    shapes = [
        gl.Shape(4, 3, 19),
        gl.Shape(8, 6, 12),
        gl.Shape(4, 3, 19),
    ]

    for a, b in itertools.pairwise(shapes):
        assert a.aspect_ratio == b.aspect_ratio


def test_different_aspect_ratio() -> None:
    a = gl.Shape(1, 2, 3)
    b = gl.Shape(1, 3, 3)

    assert a.aspect_ratio != b.aspect_ratio


def test_conv2d_output_depth_non_unity_stride() -> None:
    connected = gl.ConnectedConv2D(
        gl.Input(gl.Shape(1024, 1024, 3)),
        gl.Conv2D(name="a", filter_count=17, kernel_size=5, stride=4),
    )

    actual = connected.output_shape
    expected = gl.Shape(256, 256, 17)
    assert expected == actual


def test_conv2d_output_depth_unity_stride() -> None:
    connected = gl.ConnectedConv2D(
        gl.Input(gl.Shape(1024, 1024, 3)),
        gl.Conv2D(name="a", filter_count=17, kernel_size=5, stride=1),
    )

    actual = connected.output_shape
    expected = gl.Shape(1024, 1024, 17)
    assert expected == actual
