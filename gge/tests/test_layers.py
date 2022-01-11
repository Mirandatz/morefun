import itertools

from hypothesis import example, given
import hypothesis.strategies as hs

import gge.layers as gl


@given(
    width=hs.integers(min_value=1),
    height=hs.integers(min_value=1),
    depth=hs.integers(min_value=1),
)
def test_same_object_aspect_ratio(width: int, height: int, depth: int) -> None:
    """Two equal shapes should have the same aspect ratio."""
    shape_a = gl.Shape(width, height, depth)
    shape_b = gl.Shape(width, height, depth)

    assert shape_a.aspect_ratio == shape_b.aspect_ratio


@given(
    width=hs.integers(min_value=1),
    height=hs.integers(min_value=1),
    depth=hs.integers(min_value=1),
    factor=hs.integers(min_value=2),
)
@example(width=4, height=3, depth=19, factor=2)
def test_same_aspect_ratio(width: int, height: int, depth: int, factor: int) -> None:
    """Shapes that are a factor of one another have the same aspect ratio."""
    shape_a = gl.Shape(width, height, depth)
    shape_b = gl.Shape(width * factor, height * factor, depth)

    assert shape_a.aspect_ratio == shape_b.aspect_ratio


def test_different_aspect_ratio() -> None:
    a = gl.Shape(1, 2, 3)
    b = gl.Shape(1, 3, 3)

    assert a.aspect_ratio != b.aspect_ratio


def test_conv2d_output_depth_non_unit_stride() -> None:
    connected = gl.ConnectedConv2D(
        gl.Input(gl.Shape(1024, 1024, 3)),
        gl.Conv2D(name="a", filter_count=17, kernel_size=5, stride=4),
    )

    actual = connected.output_shape
    expected = gl.Shape(256, 256, 17)
    assert expected == actual


def test_conv2d_output_depth_unit_stride() -> None:
    connected = gl.ConnectedConv2D(
        gl.Input(gl.Shape(1024, 1024, 3)),
        gl.Conv2D(name="a", filter_count=17, kernel_size=5, stride=1),
    )

    actual = connected.output_shape
    expected = gl.Shape(1024, 1024, 17)
    assert expected == actual
