import dataclasses
import enum
import fractions
import itertools
import math
import typing


@enum.unique
class MarkerType(enum.Enum):
    FORK_POINT = enum.auto()
    MERGE_POINT = enum.auto()


@dataclasses.dataclass(frozen=True)
class MarkerLayer:
    name: str
    mark_type: MarkerType

    def __post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert isinstance(self.mark_type, MarkerType)
        assert self.name


def make_fork(name: str) -> MarkerLayer:
    return MarkerLayer(name, MarkerType.FORK_POINT)


def make_merge(name: str) -> MarkerLayer:
    return MarkerLayer(name, MarkerType.MERGE_POINT)


@dataclasses.dataclass(frozen=True)
class Conv2D:
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


@dataclasses.dataclass(frozen=True)
class Conv2DTranspose:
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


@enum.unique
class PoolType(enum.Enum):
    MAX_POOLING = enum.auto()
    AVG_POOLING = enum.auto()


@dataclasses.dataclass(frozen=True)
class Pool2D:
    name: str
    pooling_type: PoolType
    stride: int

    def _post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert isinstance(self.pooling_type, PoolType)
        assert isinstance(self.stride, int)

        assert self.name
        assert self.stride > 0


@dataclasses.dataclass(frozen=True)
class BatchNorm:
    name: str

    def __post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert self.name


Layer: typing.TypeAlias = Conv2D | Pool2D | BatchNorm | MarkerLayer


def is_real_layer(layer: Layer) -> bool:
    assert isinstance(layer, Layer)
    return not isinstance(layer, MarkerLayer)


def is_fork_marker(layer: Layer) -> bool:
    return isinstance(layer, MarkerLayer) and layer.mark_type == MarkerType.FORK_POINT


def is_merge_marker(layer: Layer) -> bool:
    return isinstance(layer, MarkerLayer) and layer.mark_type == MarkerType.MERGE_POINT


@dataclasses.dataclass(frozen=True)
class Shape:
    width: int
    height: int
    depth: int

    def __post_int__(self) -> None:
        assert isinstance(self.width, int)
        assert isinstance(self.height, int)
        assert isinstance(self.depth, int)

        assert self.width >= 1
        assert self.height >= 1
        assert self.depth >= 1

    @property
    def aspect_ratio(self) -> tuple[int, int]:
        frac = fractions.Fraction(self.width, self.height)
        return (frac.numerator, frac.denominator)


@dataclasses.dataclass(frozen=True)
class Input:
    shape: Shape

    def __post_int__(self) -> None:
        assert isinstance(self.shape, Shape)

    @property
    def name(self) -> str:
        return "input"

    @property
    def output_shape(self) -> Shape:
        return self.shape


@dataclasses.dataclass(frozen=True)
class ConnectedConv2D:
    input_layer: "ConnectedLayer"
    params: Conv2D

    def __post_int__(self) -> None:
        assert isinstance(self.input_layer, ConnectedLayer)
        assert isinstance(self.params, Conv2D)

    @property
    def output_shape(self) -> Shape:
        input_shape = self.input_layer.output_shape
        params = self.params
        out_width = int(math.ceil(input_shape.width / params.stride))
        out_height = int(math.ceil(input_shape.height / params.stride))
        out_depth = self.params.filter_count
        return Shape(width=out_width, height=out_height, depth=out_depth)


@dataclasses.dataclass(frozen=True)
class ConnectedConv2DTranspose:
    input_layer: "ConnectedLayer"
    params: Conv2DTranspose

    def __post_int__(self) -> None:
        assert isinstance(self.input_layer, ConnectedLayer)
        assert isinstance(self.params, Conv2DTranspose)

    @property
    def output_shape(self) -> Shape:
        input_shape = self.input_layer.output_shape
        params = self.params
        out_width = input_shape.width * params.stride
        out_height = input_shape.height * params.stride
        out_depth = self.params.filter_count
        return Shape(width=out_width, height=out_height, depth=out_depth)


@dataclasses.dataclass(frozen=True)
class ConnectedPool2D:
    input_layer: "ConnectedLayer"
    params: Pool2D

    def __post_int__(self) -> None:
        assert isinstance(self.input_layer, ConnectedLayer)
        assert isinstance(self.params, Pool2D)

    @property
    def output_shape(self) -> Shape:
        input_shape = self.input_layer.output_shape
        params = self.params
        out_width = int(math.ceil(input_shape.width / params.stride))
        out_height = int(math.ceil(input_shape.height / params.stride))
        out_depth = input_shape.depth
        return Shape(width=out_width, height=out_height, depth=out_depth)


@dataclasses.dataclass(frozen=True)
class ConnectedBatchNorm:
    input_layer: "ConnectedLayer"
    params: BatchNorm

    def __post_int__(self) -> None:
        assert isinstance(self.input_layer, ConnectedLayer)
        assert isinstance(self.params, BatchNorm)

    @property
    def output_shape(self) -> Shape:
        shape = self.input_layer.output_shape

        # This assert is here because mypy ;-; wants (?)
        assert isinstance(shape, Shape)
        return shape


@dataclasses.dataclass(frozen=True)
class ConnectedAdd:
    inputs: tuple["ConnectedLayer", ...]

    def __post_init__(self) -> None:
        assert isinstance(self.inputs, tuple)
        for layer in self.inputs:
            assert isinstance(layer, ConnectedLayer)

        assert len(self.inputs) >= 1
        for a, b in itertools.pairwise(self.inputs):
            assert a.output_shape == b.output_shape

    @property
    def output_shape(self) -> Shape:
        shape: Shape = self.inputs[0].output_shape
        return shape


@dataclasses.dataclass(frozen=True)
class ConnectedConcatenate:
    inputs: tuple["ConnectedLayer", ...]

    def __post_init__(self) -> None:
        assert isinstance(self.inputs, tuple)
        for layer in self.inputs:
            assert isinstance(layer, ConnectedLayer)

        assert len(self.inputs) >= 1
        for a, b in itertools.pairwise(self.inputs):
            assert a.output_shape.width == b.output_shape.width
            assert a.output_shape.height == b.output_shape.height

    @property
    def output_shape(self) -> Shape:
        depths = (layer.output_shape.depth for layer in self.inputs)
        total_depth = sum(depths)

        sample_shape = self.inputs[0].output_shape
        return Shape(
            width=sample_shape.width,
            height=sample_shape.height,
            depth=total_depth,
        )


ConnectedLayer: typing.TypeAlias = (
    Input
    | ConnectedConv2D
    | ConnectedConv2DTranspose
    | ConnectedPool2D
    | ConnectedBatchNorm
    | ConnectedAdd
    | ConnectedConcatenate
)

ConnectedMergeLayer: typing.TypeAlias = ConnectedAdd | ConnectedConcatenate
