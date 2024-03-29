import attrs
import hypothesis.strategies as hs

import morefun.neural_networks.layers as gl
from morefun.name_generator import NameGenerator


@hs.composite
def conv2ds(
    draw: hs.DrawFn,
    *,
    name_gen: NameGenerator | None = None,
) -> gl.Conv2D:
    name_gen = name_gen or NameGenerator()
    return gl.Conv2D(
        name=name_gen.gen_name(gl.Conv2D),
        filter_count=draw(hs.integers(min_value=1, max_value=9)),
        kernel_size=draw(hs.integers(min_value=1, max_value=9)),
        stride=draw(hs.integers(min_value=1, max_value=9)),
    )


@hs.composite
def shapes(draw: hs.DrawFn) -> gl.Shape:
    return gl.Shape(
        width=draw(hs.integers(min_value=1, max_value=8196)),
        height=draw(hs.integers(min_value=1, max_value=8196)),
        depth=draw(hs.integers(min_value=1, max_value=8196)),
    )


@attrs.frozen
class ShapePair:
    smaller: gl.Shape
    bigger: gl.Shape
    ratio: int


@hs.composite
def shape_pairs(draw: hs.DrawFn, *, same_depth: bool = False) -> ShapePair:
    smaller = draw(shapes())
    ratio = draw(hs.integers(min_value=2, max_value=1024))
    if same_depth:
        bigger_depth = smaller.depth
    else:
        bigger_depth = draw(hs.integers(min_value=1, max_value=1024))

    bigger = gl.Shape(
        width=smaller.width * ratio,
        height=smaller.height * ratio,
        depth=bigger_depth,
    )

    return ShapePair(
        smaller=smaller,
        bigger=bigger,
        ratio=ratio,
    )


@hs.composite
def inputs(draw: hs.DrawFn) -> gl.Input:
    return gl.Input(shape=draw(shapes()))


@hs.composite
def connected_conv2ds(draw: hs.DrawFn) -> gl.ConnectedConv2D:
    return gl.ConnectedConv2D(
        input_layer=draw(inputs()),
        params=draw(conv2ds()),
    )
