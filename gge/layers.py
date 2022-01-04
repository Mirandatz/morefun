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
        assert self.mark_type in (MarkerType.FORK_POINT, MarkerType.MERGE_POINT)


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
    def aspect_ratio(self) -> fractions.Fraction:
        return fractions.Fraction(self.width, self.height)

    def __repr__(self) -> str:
        return f"{self.width, self.height, self.depth}"


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
    input_layer: "ConnectableLayer"
    params: Conv2D

    def __post_int__(self) -> None:
        assert isinstance(self.input_layer, ConnectableLayer)
        assert isinstance(self.params, Conv2D)

    @property
    def output_shape(self) -> Shape:
        input_shape = self.input_layer.output_shape
        params = self.params
        out_width = int(math.ceil(input_shape.width / params.stride))
        out_height = int(math.ceil(input_shape.height / params.stride))
        out_depth = self.params.filter_count
        return Shape(width=out_width, height=out_height, depth=out_depth)

    def __repr__(self) -> str:
        return f"{self.params.name}: out_shape=[{self.output_shape}]"


@dataclasses.dataclass(frozen=True)
class ConnectedConv2DTranspose:
    input_layer: "ConnectableLayer"
    params: Conv2DTranspose

    def __post_int__(self) -> None:
        assert isinstance(self.input_layer, ConnectableLayer)
        assert isinstance(self.params, Conv2DTranspose)

    @property
    def output_shape(self) -> Shape:
        input_shape = self.input_layer.output_shape
        params = self.params
        out_width = input_shape.width * params.stride
        out_height = input_shape.height * params.stride
        out_depth = self.params.filter_count
        return Shape(width=out_width, height=out_height, depth=out_depth)

    def __repr__(self) -> str:
        return f"{self.params.name}: out_shape=[{self.output_shape}]"


@dataclasses.dataclass(frozen=True)
class ConnectedPool2D:
    input_layer: "ConnectableLayer"
    params: Pool2D

    def __post_int__(self) -> None:
        assert isinstance(self.input_layer, ConnectableLayer)
        assert isinstance(self.params, Pool2D)

    @property
    def output_shape(self) -> Shape:
        input_shape = self.input_layer.output_shape
        params = self.params
        out_width = int(math.ceil(input_shape.width / params.stride))
        out_height = int(math.ceil(input_shape.height / params.stride))
        out_depth = input_shape.depth
        return Shape(width=out_width, height=out_height, depth=out_depth)

    def __repr__(self) -> str:
        return f"{self.params.name}: out_shape=[{self.output_shape}]"


@dataclasses.dataclass(frozen=True)
class ConnectedBatchNorm:
    input_layer: "ConnectableLayer"
    params: BatchNorm

    def __post_int__(self) -> None:
        assert isinstance(self.input_layer, ConnectableLayer)
        assert isinstance(self.params, BatchNorm)

    @property
    def output_shape(self) -> Shape:
        shape = self.input_layer.output_shape

        # This assert is here because mypy ;-; wants (?)
        assert isinstance(shape, Shape)
        return shape

    def __repr__(self) -> str:
        return f"{self.params.name}: out_shape=[{self.output_shape}]"


@dataclasses.dataclass(frozen=True)
class ConnectedAdd:
    inputs: tuple["ConnectableLayer", ...]

    def __post_init__(self) -> None:
        assert isinstance(self.inputs, tuple)
        for layer in self.inputs:
            assert isinstance(layer, ConnectableLayer)

        assert len(self.inputs) >= 1

        shapes = (layer.output_shape for layer in self.inputs)
        if len(set(shapes)) != 1:
            raise ValueError("inputs must have the same shape")

        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("inputs must not contain duplicated layers")

    @property
    def output_shape(self) -> Shape:
        shape: Shape = self.inputs[0].output_shape
        return shape

    def __repr__(self) -> str:
        return f"add: out_shape=[{self.output_shape}]"


@dataclasses.dataclass(frozen=True)
class ConnectedConcatenate:
    inputs: tuple["ConnectableLayer", ...]

    def __post_init__(self) -> None:
        assert isinstance(self.inputs, tuple)
        for layer in self.inputs:
            assert isinstance(layer, ConnectableLayer)

        assert len(self.inputs) >= 1
        for a, b in itertools.pairwise(self.inputs):
            assert a.output_shape.width == b.output_shape.width
            assert a.output_shape.height == b.output_shape.height

        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("inputs must not contain duplicated layers")

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

    def __repr__(self) -> str:
        return f"concatenate: out_shape=[{self.output_shape}]"


# An example of another (not yet implemented) is the Constant
NoInputLayer: typing.TypeAlias = Input

SingleInputLayer: typing.TypeAlias = (
    ConnectedConv2D | ConnectedConv2DTranspose | ConnectedPool2D | ConnectedBatchNorm
)

MultiInputLayer: typing.TypeAlias = ConnectedAdd | ConnectedConcatenate


ConnectableLayer: typing.TypeAlias = NoInputLayer | SingleInputLayer | MultiInputLayer


def iter_sources(layer: ConnectableLayer) -> typing.Iterable[ConnectableLayer]:
    if isinstance(layer, NoInputLayer):
        return []

    elif isinstance(layer, SingleInputLayer):
        return [layer.input_layer]

    elif isinstance(layer, MultiInputLayer):
        return layer.inputs

    else:
        raise ValueError(f"unknown layer type: {layer}")
