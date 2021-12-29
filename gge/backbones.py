import collections
import dataclasses
import enum
import functools
import itertools
import pathlib
import typing

import lark

import gge.name_generator
import gge.transformers as gge_transformers

MESAGRAMMAR_PATH = pathlib.Path(__file__).parent.parent / "data" / "mesagrammar.lark"


@functools.cache
def get_mesagrammar() -> str:
    return MESAGRAMMAR_PATH.read_text()


@dataclasses.dataclass(frozen=True)
class Fork:
    name: str

    def __post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert self.name


@dataclasses.dataclass(frozen=True)
class Merge:
    name: str

    def __post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert self.name


@dataclasses.dataclass(frozen=True)
class Conv2DLayer:
    name: str
    filter_count: int
    kernel_size: int
    stride: int

    def __post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert isinstance(self.filter_count, int)
        assert isinstance(self.kernel_size, int)
        assert isinstance(self.stride, int)

        assert self.name
        assert self.filter_count > 0
        assert self.kernel_size > 0
        assert self.stride > 0


class PoolingType(enum.Enum):
    MAX_POOLING = enum.auto()
    AVG_POOLING = enum.auto()


@dataclasses.dataclass(frozen=True)
class PoolingLayer:
    name: str
    pooling_type: PoolingType
    stride: int

    def _post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert isinstance(self.pooling_type, PoolingType)
        assert isinstance(self.stride, int)

        assert self.name
        assert self.stride > 0


@dataclasses.dataclass(frozen=True)
class BatchNorm:
    name: str

    def __post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert self.name


Layer: typing.TypeAlias = Conv2DLayer | PoolingLayer | BatchNorm | Fork | Merge


def _raise_if_contains_sequence_of_forks(layers: tuple[Layer, ...]) -> None:
    for a, b in itertools.pairwise(layers):
        if isinstance(a, Fork) and isinstance(b, Fork):
            raise ValueError("layers can not contain sequences of forks")


def _raise_if_contains_repeated_names(layers: tuple[Layer, ...]) -> None:
    names = collections.Counter(layer.name for layer in layers)
    repeated = [name for name, times in names.items() if times > 1]
    if repeated:
        raise ValueError(
            "layers must have unique names, "
            f"but the following are repeated: {repeated}"
        )


def _raise_if_contains_repeated_names(layers: tuple[Layer, ...]) -> None:
    names = collections.Counter(layer.name for layer in layers)
    repeated = [name for name, times in names.items() if times > 1]
    if repeated:
        raise ValueError(
            f"layers must have unique names, but the following are repeated: {repeated}"
        )


def _raise_if_contains_sequences_of_forks(layers: tuple[Layer, ...]) -> None:
    previous, *other = layers

    for current in other:
        if isinstance(current, Fork) and isinstance(previous, Fork):
            raise ValueError("backbone is must not contain sequences of forks")

        previous = current


@dataclasses.dataclass(frozen=True)
class Backbone:
    layers: tuple[Layer, ...]

    def __post_init__(self) -> None:
        assert isinstance(self.layers, tuple)
        for layer in self.layers:
            assert isinstance(layer, Layer)

        len(self.layers) >= 1
        _raise_if_contains_sequence_of_forks(self.layers)
        _raise_if_contains_repeated_names(self.layers)


@lark.v_args(inline=True)
class BackboneSynthetizer(gge_transformers.SinglePassTransformer[Backbone]):
    def __init__(self) -> None:
        super().__init__()
        self._name_generator = gge.name_generator.NameGenerator()

    def _create_layer_name(self, prefix: str) -> str:
        self._raise_if_not_running()
        return self._name_generator.create_name(prefix)

    @lark.v_args(inline=False)
    def start(self, blocks: list[list[Layer]]) -> Backbone:
        self._raise_if_not_running()

        layers = tuple(layer for list_of_layers in blocks for layer in list_of_layers)
        return Backbone(layers)

    @lark.v_args(inline=False)
    def block(self, parts: typing.Any) -> list[Layer]:
        self._raise_if_not_running()

        return [x for x in parts if x is not None]

    def MERGE(self, token: lark.Token | None = None) -> Merge | None:
        self._raise_if_not_running()

        if token is not None:
            return Merge(self._create_layer_name("merge"))
        else:
            return None

    def FORK(self, token: lark.Token | None = None) -> Fork | None:
        self._raise_if_not_running()

        if token is not None:
            return Fork(self._create_layer_name("fork"))
        else:
            return None

    def layer(self, layer: Layer) -> Layer:
        self._raise_if_not_running()

        return layer

    def conv_layer(
        self, filter_count: int, kernel_size: int, stride: int
    ) -> Conv2DLayer:
        self._raise_if_not_running()

        return Conv2DLayer(
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

    def batchnorm_layer(self) -> BatchNorm:
        self._raise_if_not_running()
        return BatchNorm(self._create_layer_name("batchnorm"))

    def pool_layer(self, pool_type: PoolingType, stride: int) -> PoolingLayer:
        self._raise_if_not_running()
        return PoolingLayer(
            name=self._create_layer_name("pooling_layer"),
            pooling_type=pool_type,
            stride=stride,
        )

    def POOL_MAX(self, _: lark.Token) -> PoolingType:
        self._raise_if_not_running()
        return PoolingType.MAX_POOLING

    def POOL_AVG(self, _: lark.Token) -> PoolingType:
        self._raise_if_not_running()
        return PoolingType.AVG_POOLING

    def POOL_STRIDE(self, token: lark.Token) -> int:
        self._raise_if_not_running()
        return int(token.value)

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
