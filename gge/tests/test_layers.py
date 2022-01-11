from hypothesis import assume, example, given
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


@given(
    base_side=hs.integers(min_value=1),
    changing_side_a=hs.integers(min_value=1),
    changing_side_b=hs.integers(min_value=1),
    depth=hs.integers(min_value=1),
)
@example(base_side=1, changing_side_a=2, changing_side_b=3, depth=3)
def test_different_aspect_ratio(
    base_side: int, changing_side_a: int, changing_side_b: int, depth: int
) -> None:
    """Shapes with non-multiple values in one side do not have the same aspect ratio."""
    assume(changing_side_b % changing_side_a != 0)
    assume(changing_side_a % changing_side_b != 0)

    wide_a = gl.Shape(changing_side_a, base_side, depth)
    wide_b = gl.Shape(changing_side_b, base_side, depth)

    assert wide_a.aspect_ratio != wide_b.aspect_ratio

    high_a = gl.Shape(base_side, changing_side_a, depth)
    high_b = gl.Shape(base_side, changing_side_b, depth)

    assert high_a.aspect_ratio != high_b.aspect_ratio


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
