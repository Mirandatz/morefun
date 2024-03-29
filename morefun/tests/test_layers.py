import hypothesis
import hypothesis.strategies as hs
from hypothesis import assume, example, given

import morefun.neural_networks.layers as gl
import morefun.tests.strategies.layers as ls

# autouse fixtures
from morefun.tests.fixtures import hide_gpu_from_tensorflow, remove_logger_sinks  # noqa


@given(
    width=hs.integers(min_value=1),
    height=hs.integers(min_value=1),
    depth=hs.integers(min_value=1),
)
def test_same_object_aspect_ratio(width: int, height: int, depth: int) -> None:
    """Two equal shapes should have the same aspect ratio."""
    shape_a = gl.Shape(height=height, width=width, depth=depth)
    shape_b = gl.Shape(height=height, width=width, depth=depth)

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
    shape_a = gl.Shape(height=height, width=width, depth=depth)
    shape_b = gl.Shape(
        height=height * factor,
        width=width * factor,
        depth=depth,
    )

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

    wide_a = gl.Shape(height=base_side, width=changing_side_a, depth=depth)
    wide_b = gl.Shape(height=base_side, width=changing_side_b, depth=depth)

    assert wide_a.aspect_ratio != wide_b.aspect_ratio

    high_a = gl.Shape(height=changing_side_a, width=base_side, depth=depth)
    high_b = gl.Shape(height=changing_side_b, width=base_side, depth=depth)

    assert high_a.aspect_ratio != high_b.aspect_ratio


# max values are needed to avoid rounding above max_int
@given(
    output_width=hs.integers(min_value=1, max_value=999999),
    output_height=hs.integers(min_value=1, max_value=999999),
    depth=hs.integers(min_value=1),
    stride=hs.integers(min_value=1, max_value=999999),
    filter_count=hs.integers(min_value=1),
    kernel_size=hs.integers(min_value=1),
    layer_name=hs.text(min_size=1),
)
@example(
    output_width=256,
    output_height=256,
    depth=3,
    stride=4,
    filter_count=17,
    kernel_size=5,
    layer_name="a",
)
@example(
    output_width=1024,
    output_height=1024,
    depth=3,
    stride=1,
    filter_count=17,
    kernel_size=5,
    layer_name="a",
)
def test_conv2d_output_shape(
    output_width: int,
    output_height: int,
    depth: int,
    stride: int,
    filter_count: int,
    kernel_size: int,
    layer_name: str,
) -> None:
    """A Conv2D layer output shape is based on its stride and the input layer."""
    input_width = output_width * stride
    input_height = output_height * stride

    connected = gl.ConnectedConv2D(
        gl.Input(gl.Shape(height=input_height, width=input_width, depth=depth)),
        gl.Conv2D(
            name=layer_name,
            filter_count=filter_count,
            kernel_size=kernel_size,
            stride=stride,
        ),
    )

    actual = connected.output_shape
    expected = gl.Shape(height=output_height, width=output_width, depth=filter_count)
    assert expected == actual


@hypothesis.settings(deadline=1000)
@given(ls.connected_conv2ds())
def test_conv2d_tensor_shape(layer: gl.ConnectedConv2D) -> None:
    """Output shape of ConnectedConv2D layer should match the tensor equivalent."""

    tensor_shape = layer.to_tensor({}).shape
    _, expected_width, expected_height, expected_depth = tensor_shape

    actual_shape = layer.output_shape

    assert expected_width == actual_shape.width
    assert expected_height == actual_shape.height
    assert expected_depth == actual_shape.depth
