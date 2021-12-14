import itertools
import typing
from dataclasses import dataclass
from typing import Any, Union

import lark


@dataclass(frozen=True)
class Input:
    pass


@dataclass(frozen=True)
class Conv2d:
    filter_count: int
    kernel_size: int
    stride: int

    def __post_init__(self) -> None:
        assert self.filter_count > 0
        assert self.kernel_size > 0
        assert self.stride > 0


@dataclass(frozen=True)
class Dense:
    units: int

    def __post_init__(self) -> None:
        assert self.units > 0


@dataclass(frozen=True)
class Dropout:
    rate: float

    def __post_init__(self) -> None:
        assert 0 <= self.rate <= 1


Layer = Union[Input, Conv2d, Dense, Dropout]
MainPath = typing.Tuple[Layer, ...]


class Synthetizer(lark.Transformer[MainPath]):
    def __init__(self) -> None:
        super().__init__()

    def __default__(self, data: Any, children: Any, meta: Any) -> None:
        raise NotImplementedError(f"method not implemented for tree.data: {data}")

    def __default_token__(self, token_text: str) -> None:
        raise NotImplementedError(
            f"method not implemented for token with text: {token_text}"
        )

    def start(self, layers: list[Layer]) -> Network:
        return tuple(layers)

    def filter_count(self, data: Any) -> list[int]:
        marker, *counts = data
        return typing.cast(list[int], counts)

    def kernel_size(self, data: Any) -> list[int]:
        marker, *sizes = data
        return typing.cast(list[int], sizes)

    def stride(self, data: Any) -> list[int]:
        marker, *strides = data
        return typing.cast(list[int], strides)

    def conv_layer(self, data: Any) -> list[Conv2d]:
        marker, filter_sizes, kernel_sizes, strides = data

        conv_layers = []
        params = itertools.product(filter_sizes, kernel_sizes, strides)
        conv_layers = [
            Conv2d(
                filter_count=fs,
                kernel_size=ks,
                stride=st,
            )
            for (fs, ks, st) in params
        ]

        return conv_layers

    def dense_layer(self, data: tuple[str, str]) -> DenseLayer:
        marker, arg = data
        return DenseLayer(LayerSize(int(arg)))

    def dropout_layer(self, data: tuple[str, str]) -> DropoutLayer:
        marker, arg = data
        return DropoutLayer(Probability(float(arg)))

    def layer(self, layer: tuple[NetworkNode, ...]) -> NetworkNode:
        # layer always has exactly one node children
        return layer[0]


def main() -> None:
    pass


if __name__ == "__main__":
    main()
