import typing
from dataclasses import dataclass
from typing import Any, Union

import lark
import typeguard


@dataclass(frozen=True)
class ForkLayer:
    pass


@dataclass(frozen=True)
class MergeLayer:
    pass


@typeguard.typechecked
@dataclass(frozen=True)
class Conv2DLayer:
    filter_count: int
    kernel_size: int
    stride: int

    def __post_init__(self) -> None:
        assert self.filter_count > 0
        assert self.kernel_size > 0
        assert self.stride > 0


@typeguard.typechecked
@dataclass(frozen=True)
class DenseLayer:
    units: int

    def __post_init__(self) -> None:
        assert self.units > 0


@typeguard.typechecked
@dataclass(frozen=True)
class DropoutLayer:
    rate: float

    def __post_init__(self) -> None:
        assert 0 <= self.rate <= 1


Layer = Union[Conv2DLayer, DenseLayer, DropoutLayer, ForkLayer, MergeLayer]
Backbone = typing.Tuple[Layer, ...]


@lark.v_args(inline=True)
class BackboneSynthetizer(lark.Transformer[typing.Tuple[Layer, ...]]):
    def __init__(self) -> None:
        super().__init__()

    def __default__(self, data: Any, children: Any, meta: Any) -> None:
        raise NotImplementedError(f"method not implemented for tree.data: {data}")

    def __default_token__(self, token_text: str) -> None:
        raise NotImplementedError(
            f"method not implemented for token with text: {token_text}"
        )

    def start(self, *lists_of_layers: list[Layer]) -> tuple[Layer, ...]:
        return tuple((layer for sublist in lists_of_layers for layer in sublist))

    def layer(self, merge: bool, layer: Layer, fork: bool) -> list[Layer]:
        layers: list[Layer] = []

        if merge:
            layers.append(MergeLayer())

        layers.append(layer)

        if fork:
            layers.append(ForkLayer())

        return layers

    def maybe_merge(self, token: lark.Token | None = None) -> bool:
        return token is not None

    def maybe_fork(self, token: lark.Token | None = None) -> bool:
        return token is not None

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
