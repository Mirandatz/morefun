import collections
import dataclasses
import functools
import itertools
import pathlib
import typing

import lark

import gge.layers as gl
import gge.name_generator
import gge.transformers as gge_transformers

MESAGRAMMAR_PATH = pathlib.Path(__file__).parent.parent / "data" / "mesagrammar.lark"


@functools.cache
def get_mesagrammar() -> str:
    return MESAGRAMMAR_PATH.read_text()


def _raise_if_contains_repeated_names(layers: tuple[gl.Layer, ...]) -> None:
    names = collections.Counter(layer.name for layer in layers)
    repeated = [name for name, times in names.items() if times > 1]
    if repeated:
        raise ValueError(
            f"layers must have unique names, but the following are repeated: {repeated}"
        )


def _raise_if_contains_sequences_of_forks(layers: tuple[gl.Layer, ...]) -> None:
    for prev, curr in itertools.pairwise(layers):
        if gl.is_fork_marker(prev) and gl.is_fork_marker(curr):
            raise ValueError("backbone must not contain sequences of forks")


@dataclasses.dataclass(frozen=True)
class Backbone:
    layers: tuple[gl.Layer, ...]

    def __post_init__(self) -> None:
        assert isinstance(self.layers, tuple)
        for layer in self.layers:
            assert isinstance(layer, gl.Layer)

        real_layers = filter(gl.is_real_layer, self.layers)
        if not any(real_layers):
            raise ValueError("layers must contain at least one real layer")

        if len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must not contain duplicates")

        _raise_if_contains_sequences_of_forks(self.layers)
        _raise_if_contains_repeated_names(self.layers)


@lark.v_args(inline=True)
class BackboneSynthetizer(gge_transformers.SinglePassTransformer[Backbone]):
    def __init__(self) -> None:
        super().__init__()
        self._name_generator = gge.name_generator.NameGenerator()

    def _create_layer_name(self, prefix: str) -> str:
        self._raise_if_not_running()
        return self._name_generator.gen_name(prefix)

    @lark.v_args(inline=False)
    def start(self, blocks: list[list[gl.Layer]]) -> Backbone:
        self._raise_if_not_running()

        layers = tuple(layer for list_of_layers in blocks for layer in list_of_layers)
        return Backbone(layers)

    @lark.v_args(inline=False)
    def block(self, parts: typing.Any) -> list[gl.Layer]:
        self._raise_if_not_running()

        return [x for x in parts if x is not None]

    def MERGE(self, token: lark.Token | None = None) -> gl.MarkerLayer | None:
        self._raise_if_not_running()

        if token is not None:
            return gl.make_merge(self._create_layer_name("merge"))
        else:
            return None

    def FORK(self, token: lark.Token | None = None) -> gl.MarkerLayer | None:
        self._raise_if_not_running()

        if token is not None:
            return gl.make_fork(self._create_layer_name("fork"))
        else:
            return None

    def layer(self, layer: gl.Layer) -> gl.Layer:
        self._raise_if_not_running()

        return layer

    def conv_layer(self, filter_count: int, kernel_size: int, stride: int) -> gl.Conv2D:
        self._raise_if_not_running()

        return gl.Conv2D(
            name=self._create_layer_name("conv2d"),
            filter_count=filter_count,
            kernel_size=kernel_size,
            stride=stride,
        )

    def filter_count(self, count: int) -> int:
        self._raise_if_not_running()
        assert count >= 1

        return count

    def kernel_size(self, stride: int) -> int:
        self._raise_if_not_running()
        assert stride >= 1

        return stride

    def stride(self, size: int) -> int:
        self._raise_if_not_running()
        assert size >= 1

        return size

    def batchnorm_layer(self) -> gl.BatchNorm:
        self._raise_if_not_running()
        return gl.BatchNorm(self._create_layer_name("batchnorm"))

    def pool_layer(self, pool_type: gl.PoolType, stride: int) -> gl.Pool2D:
        self._raise_if_not_running()
        return gl.Pool2D(
            name=self._create_layer_name("pooling_layer"),
            pooling_type=pool_type,
            stride=stride,
        )

    def activation_layer(self, layer: gl.SingleInputLayer) -> gl.SingleInputLayer:
        self._raise_if_not_running()
        return layer

    def POOL_MAX(self, _: lark.Token) -> gl.PoolType:
        self._raise_if_not_running()
        return gl.PoolType.MAX_POOLING

    def POOL_AVG(self, _: lark.Token) -> gl.PoolType:
        self._raise_if_not_running()
        return gl.PoolType.AVG_POOLING

    def POOL_STRIDE(self, token: lark.Token) -> int:
        self._raise_if_not_running()
        return int(token.value)

    def RELU(self, token: lark.Token) -> gl.Relu:
        self._raise_if_not_running()
        return gl.Relu(name=self._create_layer_name("relu"))

    def GELU(self, token: lark.Token) -> gl.Gelu:
        self._raise_if_not_running()
        return gl.Gelu(name=self._create_layer_name("gelu"))

    def SWISH(self, token: lark.Token) -> gl.Swish:
        self._raise_if_not_running()
        return gl.Swish(name=self._create_layer_name("swish"))

    def INT(self, token: lark.Token) -> int:
        self._raise_if_not_running()
        return int(token.value)

    def FLOAT(self, token: lark.Token) -> float:
        self._raise_if_not_running()
        return float(token.value)


def parse(string: str) -> Backbone:
    """
    This is not a "string deserialization function";
    the input string is expected to be a "token stream"
    that can be translated into an abstract syntax tree that can
    be visited/transformed into a `Backbone`.
    """

    parser = lark.Lark(grammar=get_mesagrammar(), parser="lalr")
    tree = parser.parse(string)
    return BackboneSynthetizer().transform(tree)
