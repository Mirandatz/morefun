import dataclasses
import typing

import lark
import typeguard


@dataclasses.dataclass(frozen=True)
class Fork:
    pass


@dataclasses.dataclass(frozen=True)
class Merge:
    pass


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Conv2DLayer:
    filter_count: int
    kernel_size: int
    stride: int

    def __post_init__(self) -> None:
        assert self.filter_count > 0
        assert self.kernel_size > 0
        assert self.stride > 0


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class DenseLayer:
    units: int

    def __post_init__(self) -> None:
        assert self.units > 0


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class DropoutLayer:
    rate: float

    def __post_init__(self) -> None:
        assert 0 <= self.rate <= 1


Layer = Conv2DLayer | DenseLayer | DropoutLayer | Fork | Merge


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Backbone:
    layers: tuple[Layer, ...]


@lark.v_args(inline=True)
class BackboneSynthetizer(lark.Transformer[Backbone]):
    def __init__(self) -> None:
        super().__init__()

    def __default__(
        self, data: typing.Any, children: typing.Any, meta: typing.Any
    ) -> None:
        raise NotImplementedError(f"method not implemented for tree.data: {data}")

    def __default_token__(self, token_text: str) -> None:
        raise NotImplementedError(
            f"method not implemented for token with text: {token_text}"
        )

    @lark.v_args(inline=False)
    def start(self, blocks: list[list[Layer]]) -> Backbone:
        layers = tuple(layer for list_of_layers in blocks for layer in list_of_layers)
        return Backbone(layers)

    @lark.v_args(inline=False)
    def block(self, parts: typing.Any) -> list[Layer]:
        return [x for x in parts if x is not None]

    def MERGE(self, token: lark.Token | None = None) -> Merge | None:
        if token is not None:
            return Merge()
        else:
            return None

    def FORK(self, token: lark.Token | None = None) -> Fork | None:
        if token is not None:
            return Fork()
        else:
            return None

    def layer(self, layer: Layer) -> Layer:
        return layer

    def conv_layer(
        self, filter_count: int, kernel_size: int, stride: int
    ) -> Conv2DLayer:
        return Conv2DLayer(
            filter_count=filter_count,
            kernel_size=kernel_size,
            stride=stride,
        )

    def filter_count(self, count: int) -> int:
        assert count >= 1
        return count

    def kernel_size(self, stride: int) -> int:
        assert stride >= 1
        return stride

    def stride(self, size: int) -> int:
        assert size >= 1
        return size

    def dense_layer(self, units: int) -> DenseLayer:
        assert units >= 1
        return DenseLayer(units)

    def dropout_layer(self, rate: float) -> DropoutLayer:
        assert 0 < rate < 1
        return DropoutLayer(rate)

    def INT(self, token: lark.Token) -> int:
        return int(token.value)

    def FLOAT(self, token: lark.Token) -> float:
        return float(token.value)
