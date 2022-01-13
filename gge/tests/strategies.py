import dataclasses
import operator
import typing
from functools import reduce

import hypothesis.strategies as hs

import gge.layers as gl
import gge.name_generator


@dataclasses.dataclass(frozen=True)
class GrammarOption:
    """the mesagrammar '(1 | 2 | 3)' string has values [1, 2, 3]"""

    possible_values: list[int]

    @property
    def mesagrammar_string(self) -> str:
        return f"({'|'.join(map(str, self.possible_values))})"


@dataclasses.dataclass(frozen=True)
class GrammarLayer:
    """pair of layers and respective mesagrammar string"""

    layers: tuple[gl.Layer, ...]
    mesagrammar_string: str


@hs.composite
def grammar_integer_option(draw: typing.Callable[..., typing.Any]) -> GrammarOption:
    ints = draw(hs.lists(hs.integers(min_value=1), min_size=2, unique=True))
    return GrammarOption(ints)


@hs.composite
def with_markers(
    draw: typing.Callable[..., typing.Any], element: GrammarLayer
) -> GrammarLayer:
    """randomly adds merge and fork points to a layer

    This should shrink towards having no markers."""
    name_gen = gge.name_generator.NameGenerator()
    merge_name = name_gen.gen_name("merge")
    fork_name = name_gen.gen_name("fork")

    merge = draw(hs.booleans())
    fork = draw(hs.booleans())
    merge_layer = gl.make_merge(merge_name)
    fork_layer = gl.make_fork(fork_name)

    layers: tuple[gl.Layer, ...] = element.layers
    mesa_str = element.mesagrammar_string
    if merge:
        layers = (merge_layer,) + layers
        mesa_str = '"merge"' + mesa_str
    if fork:
        layers = layers + (fork_layer,)
        mesa_str = mesa_str + '"fork"'

    return GrammarLayer(layers, mesa_str)


def temporary_name(cls: type) -> str:
    """used for single-use calls of NameGenerator"""
    gen = gge.name_generator.NameGenerator()
    return gen.gen_name(cls)


@hs.composite
def conv2d_layer(
    draw: typing.Callable[..., typing.Any],
) -> GrammarLayer:
    name = temporary_name(gl.Conv2D)

    filter_count = draw(hs.integers(min_value=1))
    kernel_size = draw(hs.integers(min_value=1))
    stride = draw(hs.integers(min_value=1))

    conv_layer = gl.Conv2D(name, filter_count, kernel_size, stride)
    conv_str = f'"conv2d" "filter_count" {filter_count} "kernel_size" {kernel_size} "stride" {stride}'

    return GrammarLayer(layers=(conv_layer,), mesagrammar_string=conv_str)


@hs.composite
def pool_layer(draw: typing.Callable[..., typing.Any]) -> GrammarLayer:
    name = temporary_name(gl.Pool2D)
    pool_type = draw(hs.sampled_from(gl.PoolType))
    stride = draw(hs.integers(min_value=1, max_value=999999))
    layer = gl.Pool2D(name, pool_type, stride)

    pool_name = {gl.PoolType.MAX_POOLING: "max", gl.PoolType.AVG_POOLING: "avg"}[
        pool_type
    ]
    mesa_str = f'"pool2d" "{pool_name}" {stride}'

    return GrammarLayer(layers=(layer,), mesagrammar_string=mesa_str)


@hs.composite
def batch_norm_layer(draw: typing.Callable[..., typing.Any]) -> GrammarLayer:
    name = temporary_name(gl.BatchNorm)
    return GrammarLayer(layers=(gl.BatchNorm(name),), mesagrammar_string='"batchnorm"')


@hs.composite
def relu_layer(draw: typing.Callable[..., typing.Any]) -> GrammarLayer:
    name = temporary_name(gl.Relu)
    return GrammarLayer(layers=(gl.Relu(name),), mesagrammar_string='"relu"')


@hs.composite
def gelu_layer(draw: typing.Callable[..., typing.Any]) -> GrammarLayer:
    name = temporary_name(gl.Gelu)
    return GrammarLayer(layers=(gl.Gelu(name),), mesagrammar_string='"gelu"')


@hs.composite
def swish_layer(draw: typing.Callable[..., typing.Any]) -> GrammarLayer:
    name = temporary_name(gl.Swish)
    return GrammarLayer(layers=(gl.Swish(name),), mesagrammar_string='"swish"')


@hs.composite
def any_layer(
    draw: typing.Callable[..., GrammarLayer],
    *,
    can_mark: bool = True,
    valid_layers: list[typing.Callable[..., hs.SearchStrategy[GrammarLayer]]] = [
        conv2d_layer,
        pool_layer,
        batch_norm_layer,
        relu_layer,
        gelu_layer,
        swish_layer,
    ],
) -> GrammarLayer:
    layer = draw(hs.sampled_from([draw(valid()) for valid in valid_layers]))
    if can_mark:
        layer = draw(with_markers(layer))
    return layer


def uniquely_named(backbone: GrammarLayer) -> GrammarLayer:
    """replaces names of a layer sequence for unique generated names"""
    gen = gge.name_generator.NameGenerator()

    def get_name(layer: gl.Layer) -> str:
        if gl.is_merge_marker(layer):
            return gen.gen_name("merge")
        if gl.is_fork_marker(layer):
            return gen.gen_name("fork")
        return gen.gen_name(type(layer))

    new_layers = (
        dataclasses.replace(layer, name=get_name(layer)) for layer in backbone.layers
    )
    return GrammarLayer(
        layers=tuple(new_layers), mesagrammar_string=backbone.mesagrammar_string
    )


@hs.composite
def backbone(
    draw: typing.Callable[..., typing.Any],
    *,
    min_size: int = 1,
    max_size: int = 10,
    valid_layers: list[typing.Callable[..., hs.SearchStrategy[GrammarLayer]]]
    | None = None,
) -> GrammarLayer:
    layer_strat = (
        any_layer() if valid_layers is None else any_layer(valid_layers=valid_layers)
    )
    grammar_layers: list[GrammarLayer] = draw(
        hs.lists(
            layer_strat,
            min_size=min_size,
            max_size=max_size,
        )
    )
    flattened_layers = reduce(operator.add, [layer.layers for layer in grammar_layers])
    concat_mesa_str = reduce(
        operator.add, [layer.mesagrammar_string for layer in grammar_layers]
    )

    base_backbone = GrammarLayer(flattened_layers, concat_mesa_str)
    return uniquely_named(base_backbone)
