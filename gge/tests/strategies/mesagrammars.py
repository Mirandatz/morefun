import attrs
import hypothesis.strategies as hs

import gge.layers as gl
import gge.optimizers as optim
from gge.name_generator import NameGenerator


@attrs.frozen
class LayersTestData:
    token_stream: str
    parsed: tuple[gl.Layer, ...]


@hs.composite
def conv2ds(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LayersTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.Conv2D)
    filter_count = draw(hs.integers(min_value=1))
    kernel_size = draw(hs.integers(min_value=1))
    stride = draw(hs.integers(min_value=1))
    layer = gl.Conv2D(name, filter_count, kernel_size, stride)
    token_stream = f'"conv2d" "filter_count" {filter_count} "kernel_size" {kernel_size} "stride" {stride}'
    return LayersTestData(token_stream, (layer,))


@hs.composite
def max_pool2ds(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LayersTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.MaxPool2D)
    pool_size = draw(hs.integers(min_value=1, max_value=9999))
    stride = draw(hs.integers(min_value=1, max_value=9999))
    layer = gl.MaxPool2D(name, pool_size=pool_size, stride=stride)
    token_stream = f'"max_pool2d" "pool_size" {pool_size} "stride" {stride}'
    return LayersTestData(token_stream, (layer,))


@hs.composite
def avg_pool2ds(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LayersTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.AvgPool2D)
    pool_size = draw(hs.integers(min_value=1, max_value=9999))
    stride = draw(hs.integers(min_value=1, max_value=9999))
    layer = gl.AvgPool2D(name, pool_size=pool_size, stride=stride)
    token_stream = f'"avg_pool2d" "pool_size" {pool_size} "stride" {stride}'
    return LayersTestData(token_stream, (layer,))


@hs.composite
def batchnorms(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LayersTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.BatchNorm)
    return LayersTestData(
        token_stream='"batchnorm"',
        parsed=(gl.BatchNorm(name),),
    )


@hs.composite
def relus(draw: hs.DrawFn, *, name_gen: NameGenerator | None = None) -> LayersTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.Relu)
    return LayersTestData(
        token_stream='"relu"',
        parsed=(gl.Relu(name),),
    )


@hs.composite
def gelus(draw: hs.DrawFn, *, name_gen: NameGenerator | None = None) -> LayersTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.Gelu)
    return LayersTestData(
        token_stream='"gelu"',
        parsed=(gl.Gelu(name),),
    )


@hs.composite
def swishs(draw: hs.DrawFn, *, name_gen: NameGenerator | None = None) -> LayersTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.Swish)
    return LayersTestData(
        token_stream='"swish"',
        parsed=(gl.Swish(name),),
    )


@hs.composite
def merge_marker(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LayersTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name("merge")
    return LayersTestData(
        token_stream='"merge"',
        parsed=(gl.make_merge(name),),
    )


@hs.composite
def fork_marker(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LayersTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name("fork")
    return LayersTestData(
        token_stream='"fork"',
        parsed=(gl.make_fork(name),),
    )


@hs.composite
def add_markers(
    draw: hs.DrawFn,
    strat: hs.SearchStrategy[LayersTestData],
    *,
    name_gen: NameGenerator | None = None,
) -> LayersTestData:
    name_gen = name_gen or NameGenerator()
    raise NotImplementedError()
    # add_merge = draw(hs.booleans())
    # if add_merge:
    #     merge_name = name_gen.gen_name('merge')


@hs.composite
def add_dummy_optimizer_to(
    draw: hs.DrawFn,
    strat: hs.SearchStrategy[LayersTestData],
) -> LayersTestData:
    test_data = draw(strat)
    dummy_optimizer = '"sgd" "learning_rate" 0.1 "momentum" 0.05 "nesterov" false'
    updated_tokenstream = test_data.token_stream + dummy_optimizer
    return attrs.evolve(test_data, token_stream=updated_tokenstream)


@attrs.frozen
class SGDTestData:
    token_stream: str
    parsed: optim.SGD


@hs.composite
def sgds(draw: hs.DrawFn) -> SGDTestData:
    learning_rate = draw(hs.floats(min_value=0, max_value=9, exclude_min=True))
    momentum = draw(hs.floats(min_value=0, max_value=9))
    nesterov = draw(hs.booleans())
    sgd = optim.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
    )
    token_stream = f'"sgd" "learning_rate" {learning_rate} "momentum" {momentum} "nesterov" {str(nesterov).lower()}'
    return SGDTestData(token_stream, sgd)


@hs.composite
def add_dummy_layer_to(
    draw: hs.DrawFn, strat: hs.SearchStrategy[SGDTestData]
) -> SGDTestData:
    test_data = draw(strat)
    dummy_layer = '"batchnorm"'
    updated_tokenstream = dummy_layer + test_data.token_stream
    return attrs.evolve(test_data, token_stream=updated_tokenstream)


# @hs.composite
# def with_markers(draw: hs.DrawFn, element: LayersTestData) -> LayersTestData:
#     """randomly adds merge and fork points to a layer

#     This should shrink towards having no markers."""
#     name_gen = gge.name_generator.NameGenerator()
#     merge_name = name_gen.gen_name("merge")
#     fork_name = name_gen.gen_name("fork")

#     merge = draw(hs.booleans())
#     fork = draw(hs.booleans())
#     merge_layer = gl.make_merge(merge_name)
#     fork_layer = gl.make_fork(fork_name)

#     layers: tuple[gl.Layer, ...] = element.layers
#     mesa_str = element.mesagrammar_string
#     if merge:
#         layers = (merge_layer,) + layers
#         mesa_str = '"merge"' + mesa_str
#     if fork:
#         layers = layers + (fork_layer,)
#         mesa_str = mesa_str + '"fork"'

#     return LayersTestData(layers, mesa_str)
