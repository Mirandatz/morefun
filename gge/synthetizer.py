import dataclasses
from typing import Any, NewType, Union

import lark


FilterCount = NewType("FilterCount", int)
KernelSize = NewType("KernelSize", int)
Stride = NewType("Stride", int)
LayerSize = NewType("LayerSize", int)
Probability = NewType("Probability", float)


@dataclasses.dataclass
class ConvLayer:
    filter_count: FilterCount
    kernel_size: KernelSize
    stride: Stride


@dataclasses.dataclass
class DenseLayer:
    layer_size: LayerSize


@dataclasses.dataclass
class DropoutLayer:
    dropout_chance: Probability


# TODO: this is a sample implementation
NetworkNode = Union[ConvLayer, DenseLayer, DropoutLayer]
Network = tuple[NetworkNode]


class Synthetizer(lark.Transformer[Network]):
    def __init__(self) -> None:
        super().__init__()

    def __default__(
        self,
        data: str,
        children: list[Any],
        meta: lark.tree.Meta,
    ) -> None:
        raise NotImplementedError(f"method not implemented for tree.data: {data}")

    def __default_token__(self, token_str: str) -> str:
        return token_str.strip('"()')

    def filter_count(self, data: tuple[str, str]) -> FilterCount:
        marker, count = data
        return FilterCount(int(count))

    def kernel_size(self, data: tuple[str, str]) -> KernelSize:
        marker, count = data
        return KernelSize(int(count))

    def stride(self, data: tuple[str, str]) -> Stride:
        marker, count = data
        return Stride(int(count))

    def conv_layer(
        self, parts: tuple[str, FilterCount, KernelSize, Stride]
    ) -> ConvLayer:
        marker, filter_size, kernel_size, stride = parts
        return ConvLayer(filter_size, kernel_size, stride)

    def dense_layer(self, data: tuple[str, str]) -> DenseLayer:
        marker, arg = data
        return DenseLayer(LayerSize(int(arg)))

    def dropout_layer(self, data: tuple[str, str]) -> DropoutLayer:
        marker, arg = data
        return DropoutLayer(Probability(float(arg)))

    def layer(self, layer: tuple[NetworkNode, ...]) -> NetworkNode:
        # layer always has exactly one node children
        return layer[0]

    def start(self, layers: list[NetworkNode]) -> Network:
        return tuple(layers)
