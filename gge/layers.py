import dataclasses
import enum
import math

import typeguard


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
class Conv2D:
    name: str
    filter_count: int
    kernel_size: int
    stride: int

    def __post_init__(self) -> None:
        assert self.name
        assert self.filter_count > 0
        assert self.kernel_size > 0
        assert self.stride > 0


class PoolType(enum.Enum):
    MAX_POOLING = enum.auto()
    AVG_POOLING = enum.auto()


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Pool2D:
    name: str
    pooling_type: PoolType
    stride: int

    def _post_init__(self) -> None:
        assert self.name
        assert self.stride > 0


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class BatchNorm:
    name: str

    def __post_init__(self) -> None:
        assert self.name


Layer = Conv2D | Pool2D | BatchNorm | Fork | Merge


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Shape:
    width: int
    height: int
    depth: int

    def __post_int__(self) -> None:
        assert self.width >= 1
        assert self.height >= 1
        assert self.depth >= 1


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Input:
    shape: Shape

    @property
    def name(self) -> str:
        return "input"

    @property
    def output_shape(self) -> Shape:
        return self.shape


@dataclasses.dataclass(frozen=True)
@typeguard.typechecked
class ConnectedConv2D:
    input_layer: "ConnectedLayer"
    params: Conv2D

    @property
    def output_shape(self) -> Shape:
        input_shape = self.input_layer.output_shape
        params = self.params
        out_width = int(math.ceil(input_shape.width / params.stride))
        out_height = int(math.ceil(input_shape.height / params.stride))
        out_depth = self.params.filter_count
        return Shape(width=out_width, height=out_height, depth=out_depth)


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class ConnectedPool2D:
    input_layer: "ConnectedLayer"
    params: Pool2D

    @property
    def output_shape(self) -> Shape:
        input_shape = self.input_layer.output_shape
        params = self.params
        out_width = int(math.ceil(input_shape.width / params.stride))
        out_height = int(math.ceil(input_shape.height / params.stride))
        out_depth = input_shape.depth
        return Shape(width=out_width, height=out_height, depth=out_depth)


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class ConnectedBatchNorm:
    input_layer: "ConnectedLayer"
    params: BatchNorm

    @property
    def output_shape(self) -> Shape:
        return self.input_layer.output_shape


ConnectedLayer = Input | ConnectedConv2D | ConnectedPool2D
