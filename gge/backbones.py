import collections
import dataclasses
import pathlib
import typing

import lark
import typeguard

MESAGRAMMAR_PATH = pathlib.Path(__file__).parent.parent / "data" / "mesagrammar.lark"
MESAGRAMMAR = MESAGRAMMAR_PATH.read_text()


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Fork:
    name: str

    def __post_init__(self) -> None:
        assert self.name


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Merge:
    name: str

    def __post_init__(self) -> None:
        assert self.name


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Conv2DLayer:
    name: str
    filter_count: int
    kernel_size: int
    stride: int

    def __post_init__(self) -> None:
        assert self.name
        assert self.filter_count > 0
        assert self.kernel_size > 0
        assert self.stride > 0


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class DenseLayer:
    name: str
    units: int

    def __post_init__(self) -> None:
        assert self.name
        assert self.units > 0


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class DropoutLayer:
    name: str
    rate: float

    def __post_init__(self) -> None:
        assert self.name
        assert 0 <= self.rate <= 1


Layer = Conv2DLayer | DenseLayer | DropoutLayer | Fork | Merge


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Backbone:
    layers: tuple[Layer, ...]

    def __post_init__(self) -> None:
        names = collections.Counter(layer.name for layer in self.layers)
        repeated = [name for name, times in names.items() if times > 1]
        if repeated:
            raise ValueError(
                f"layers must have unique names, but the following are repeated: {repeated}"
            )


@lark.v_args(inline=True)
class BackboneSynthetizer(lark.Transformer[Backbone]):
    def __init__(self) -> None:
        super().__init__()

        self._layer_counter: collections.Counter[str] = collections.Counter()
        self._is_running = False

    def transform(self, tree: lark.Tree) -> Backbone:
        assert not self._is_running

        self._is_running = True
        self._layer_counter.clear()

        ret = super().transform(tree)

        self._is_running = False

        return ret

    def __default__(
        self, data: typing.Any, children: typing.Any, meta: typing.Any
    ) -> None:
        raise NotImplementedError(f"method not implemented for tree.data: {data}")

    def __default_token__(self, token_text: str) -> None:
        raise NotImplementedError(
            f"method not implemented for token with text: {token_text}"
        )

    def _create_layer_name(self, suffix: str) -> str:
        assert self._is_running

        instance_id = self._layer_counter[suffix]
        self._layer_counter[suffix] += 1
        return f"{suffix}_{instance_id}"

    @lark.v_args(inline=False)
    def start(self, blocks: list[list[Layer]]) -> Backbone:
        assert self._is_running

        layers = tuple(layer for list_of_layers in blocks for layer in list_of_layers)
        return Backbone(layers)

    @lark.v_args(inline=False)
    def block(self, parts: typing.Any) -> list[Layer]:
        assert self._is_running

        return [x for x in parts if x is not None]

    def MERGE(self, token: lark.Token | None = None) -> Merge | None:
        assert self._is_running

        if token is not None:
            return Merge(self._create_layer_name("merge"))
        else:
            return None

    def FORK(self, token: lark.Token | None = None) -> Fork | None:
        assert self._is_running

        if token is not None:
            return Fork(self._create_layer_name("fork"))
        else:
            return None

    def layer(self, layer: Layer) -> Layer:
        assert self._is_running

        return layer

    def conv_layer(
        self, filter_count: int, kernel_size: int, stride: int
    ) -> Conv2DLayer:
        assert self._is_running

        return Conv2DLayer(
            name=self._create_layer_name("conv2d"),
            filter_count=filter_count,
            kernel_size=kernel_size,
            stride=stride,
        )

    def filter_count(self, count: int) -> int:
        assert self._is_running
        assert count >= 1

        return count

    def kernel_size(self, stride: int) -> int:
        assert self._is_running
        assert stride >= 1

        return stride

    def stride(self, size: int) -> int:
        assert self._is_running
        assert size >= 1

        return size

    def dense_layer(self, units: int) -> DenseLayer:
        assert self._is_running
        assert units >= 1

        return DenseLayer(
            name=self._create_layer_name("dense"),
            units=units,
        )

    def dropout_layer(self, rate: float) -> DropoutLayer:
        assert self._is_running
        assert 0 < rate < 1

        return DropoutLayer(
            name=self._create_layer_name("dropout"),
            rate=rate,
        )

    def INT(self, token: lark.Token) -> int:
        assert self._is_running

        return int(token.value)

    def FLOAT(self, token: lark.Token) -> float:
        assert self._is_running

        return float(token.value)


def parse(string: str) -> Backbone:
    """
    This is not a "string deserialization function";
    the input string is expected to be a "token stream"
    that can be translated into an abstract syntax tree that can
    be visited/transformed into a `Backbone`.
    """

    parser = lark.Lark(grammar=MESAGRAMMAR, parser="lalr")
    tree = parser.parse(string)
    return BackboneSynthetizer().transform(tree)
