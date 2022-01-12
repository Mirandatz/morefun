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
def conv2d(
    draw: typing.Callable[..., typing.Any],
    *,
    can_merge: bool = False,
    can_fork: bool = False,
) -> GrammarLayer:
    name_gen = gge.name_generator.NameGenerator()
    conv_name = name_gen.gen_name("conv2d")
    merge_name = name_gen.gen_name("merge")
    fork_name = name_gen.gen_name("fork")

    filter_count = draw(hs.integers(min_value=1))
    kernel_size = draw(hs.integers(min_value=1))
    stride = draw(hs.integers(min_value=1))

    merge = can_merge and draw(hs.booleans())
    fork = can_fork and draw(hs.booleans())

    merge_layer = gl.make_merge(merge_name)
    conv_layer = gl.Conv2D(conv_name, filter_count, kernel_size, stride)
    fork_layer = gl.make_fork(fork_name)

    layers: tuple[gl.Layer, ...] = (conv_layer,)
    if merge:
        layers = (merge_layer,) + layers
    if fork:
        layers = layers + (fork_layer,)

    merge_str = f"""{'"merge"' if merge else ''}"""
    fork_str = f"""{'"fork"' if fork else ''}"""
    conv_str = f'"conv2d" "filter_count" {filter_count} "kernel_size" {kernel_size} "stride" {stride}'
    mesa_str = f"{merge_str} {conv_str} {fork_str}"

    return GrammarLayer(layers, mesa_str)


@hs.composite
def uniquely_named(
    draw: typing.Callable[..., typing.Any], layers: tuple[gl.Layer, ...]
) -> tuple[gl.Layer, ...]:
    """replaces names of a layer sequence for unique generated names"""
    gen = gge.name_generator.NameGenerator()

    def get_name(layer: gl.Layer) -> str:
        if isinstance(layer, gl.Conv2D):
            return gen.gen_name("conv2d")
        if isinstance(layer, gl.MarkerLayer):
            if layer.mark_type == gl.MarkerType.MERGE_POINT:
                return gen.gen_name("merge")
            else:
                return gen.gen_name("fork")
        raise NotImplementedError()

    new_layers = (dataclasses.replace(layer, name=get_name(layer)) for layer in layers)
    return tuple(new_layers)


@hs.composite
def conv2d_backbone(
    draw: typing.Callable[..., typing.Any], min_size: int = 1, max_size: int = 1
) -> GrammarLayer:
    """creates a sequence of layers using Conv2D, merge and fork layers

    the min_size/max_size params is for the number of Conv2D, merge/fork does not count"""
    grammar_layers: list[GrammarLayer] = draw(
        hs.lists(
            conv2d(can_merge=True, can_fork=True),
            min_size=min_size,
            max_size=max_size,
        )
    )
    flattened_layers = reduce(operator.add, [layer.layers for layer in grammar_layers])
    concat_mesa_str = reduce(
        operator.add, [layer.mesagrammar_string for layer in grammar_layers]
    )

    new_layers = draw(uniquely_named(flattened_layers))

    return GrammarLayer(new_layers, concat_mesa_str)
