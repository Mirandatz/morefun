import abc
import enum
import fractions
import itertools
import math
import typing

import attrs
import keras.layers as kl
import tensorflow as tf


@enum.unique
class MarkerType(enum.Enum):
    FORK_POINT = enum.auto()
    MERGE_POINT = enum.auto()


@attrs.frozen(cache_hash=True)
class MarkerLayer:
    name: str
    mark_type: MarkerType

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert isinstance(self.mark_type, MarkerType)

        assert self.name
        assert self.mark_type in (MarkerType.FORK_POINT, MarkerType.MERGE_POINT)


def make_fork(name: str) -> MarkerLayer:
    return MarkerLayer(name, MarkerType.FORK_POINT)


def make_merge(name: str) -> MarkerLayer:
    return MarkerLayer(name, MarkerType.MERGE_POINT)


class ConvertibleToConnectableLayer(abc.ABC):
    @abc.abstractmethod
    def to_connectable(self, input: "ConnectableLayer") -> "ConnectableLayer":
        raise NotImplementedError("this is an abstract method")


@attrs.frozen(cache_hash=True)
class Conv2D(ConvertibleToConnectableLayer):
    name: str
    filter_count: int
    kernel_size: int
    stride: int

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert isinstance(self.filter_count, int)
        assert isinstance(self.kernel_size, int)
        assert isinstance(self.stride, int)

        assert self.name
        assert self.filter_count > 0
        assert self.kernel_size > 0
        assert self.stride > 0

    def to_connectable(self, input: "ConnectableLayer") -> "ConnectedConv2D":
        return ConnectedConv2D(input, self)


# MUST RENAME THIS LATER
@attrs.frozen(cache_hash=True)
class Conv2DTranspose(ConvertibleToConnectableLayer):
    name: str
    filter_count: int
    kernel_size: int
    stride: int

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert isinstance(self.filter_count, int)
        assert isinstance(self.kernel_size, int)
        assert isinstance(self.stride, int)

        assert self.name
        assert self.filter_count > 0
        assert self.kernel_size > 0
        assert self.stride > 0

    def to_connectable(self, input: "ConnectableLayer") -> "ConnectedConv2DTranspose":
        return ConnectedConv2DTranspose(input, self)


@attrs.frozen(cache_hash=True)
class MaxPool2D(ConvertibleToConnectableLayer):
    name: str
    pool_size: int
    stride: int

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert isinstance(self.pool_size, int)
        assert isinstance(self.stride, int)

        assert self.pool_size > 0
        assert self.stride > 0

    def to_connectable(self, input: "ConnectableLayer") -> "ConnectedMaxPooling2D":
        return ConnectedMaxPooling2D(input, self)


@attrs.frozen(cache_hash=True)
class AvgPool2D(ConvertibleToConnectableLayer):
    name: str
    pool_size: int
    stride: int

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert isinstance(self.pool_size, int)
        assert isinstance(self.stride, int)

        assert self.pool_size > 0
        assert self.stride > 0

    def to_connectable(self, input: "ConnectableLayer") -> "ConnectedAveragePooling2D":
        return ConnectedAveragePooling2D(input, self)


@attrs.frozen(cache_hash=True)
class BatchNorm(ConvertibleToConnectableLayer):
    name: str

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert self.name

    def to_connectable(self, input: "ConnectableLayer") -> "ConnectedBatchNorm":
        return ConnectedBatchNorm(input, self)


@attrs.frozen(cache_hash=True)
class Relu(ConvertibleToConnectableLayer):
    name: str

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert self.name

    def to_connectable(self, input: "ConnectableLayer") -> "ConnectedRelu":
        return ConnectedRelu(input, self)


@attrs.frozen(cache_hash=True)
class Gelu(ConvertibleToConnectableLayer):
    name: str

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert self.name

    def to_connectable(self, input: "ConnectableLayer") -> "ConnectedGelu":
        return ConnectedGelu(input, self)


@attrs.frozen(cache_hash=True)
class Swish(ConvertibleToConnectableLayer):
    name: str

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert self.name

    def to_connectable(self, input: "ConnectableLayer") -> "ConnectedSwish":
        return ConnectedSwish(input, self)


Layer: typing.TypeAlias = ConvertibleToConnectableLayer | MarkerLayer


def is_real_layer(layer: Layer) -> bool:
    assert isinstance(layer, Layer)
    return not isinstance(layer, MarkerLayer)


def is_fork_marker(layer: Layer) -> bool:
    return isinstance(layer, MarkerLayer) and layer.mark_type == MarkerType.FORK_POINT


def is_merge_marker(layer: Layer) -> bool:
    return isinstance(layer, MarkerLayer) and layer.mark_type == MarkerType.MERGE_POINT


@attrs.frozen(cache_hash=True)
class Shape:
    width: int
    height: int
    depth: int

    def __attrs_post_init__(self) -> None:
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


class ConnectableLayer(abc.ABC):
    @property
    @abc.abstractmethod
    def output_shape(self) -> Shape:
        raise NotImplementedError("this is an abstract method")

    @abc.abstractmethod
    def to_tensor(
        self,
        known_tensores: dict["ConnectableLayer", tf.Tensor],
    ) -> tf.Tensor:
        """
        DOCUMENT WHY THE API IS LIKE THIS.
        """
        raise NotImplementedError("this is an abstract method")

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError("this is an abstract method")


class SingleInputLayer(ConnectableLayer):
    @property
    @abc.abstractmethod
    def input_layer(self) -> ConnectableLayer:
        ...


class MultiInputLayer(ConnectableLayer):
    @property
    @abc.abstractmethod
    def input_layers(self) -> tuple[ConnectableLayer, ...]:
        ...


@attrs.frozen(cache_hash=True)
class Input(ConnectableLayer):
    NAME: typing.ClassVar[str] = "input"

    shape: Shape

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.shape, Shape)

    @property
    def name(self) -> str:
        # We rely on this being a constant, do not change!
        return Input.NAME

    @property
    def output_shape(self) -> Shape:
        return self.shape

    def to_tensor(
        self,
        known_tensores: dict["ConnectableLayer", tf.Tensor],
    ) -> tf.Tensor:
        if self not in known_tensores:
            tensor = kl.Input(
                shape=(
                    self.shape.width,
                    self.shape.height,
                    self.shape.depth,
                ),
                name=self.name,
            )
            known_tensores[self] = tensor

        return known_tensores[self]


def make_input(width: int, height: int, depth: int) -> Input:
    shape = Shape(width=width, height=height, depth=depth)
    return Input(shape)


@attrs.frozen(cache_hash=True)
class ConnectedConv2D(SingleInputLayer):
    input_layer: ConnectableLayer
    params: Conv2D

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.input_layer, ConnectableLayer)
        assert isinstance(self.params, Conv2D)

    @property
    def name(self) -> str:
        return self.params.name

    @property
    def output_shape(self) -> Shape:
        input_shape = self.input_layer.output_shape
        params = self.params
        out_width = int(math.ceil(input_shape.width / params.stride))
        out_height = int(math.ceil(input_shape.height / params.stride))
        out_depth = self.params.filter_count
        return Shape(width=out_width, height=out_height, depth=out_depth)

    def to_tensor(
        self,
        known_tensores: dict["ConnectableLayer", tf.Tensor],
    ) -> tf.Tensor:
        if self not in known_tensores:
            source = self.input_layer.to_tensor(known_tensores)

            params = self.params
            strides = (params.stride, params.stride)

            layer = kl.Conv2D(
                filters=params.filter_count,
                kernel_size=params.kernel_size,
                strides=strides,
                padding="same",
                name=params.name,
            )
            tensor = layer(source)
            known_tensores[self] = tensor

        return known_tensores[self]

    def __repr__(self) -> str:
        return f"{self.params.name}, params={self.params}, input={self.input_layer}, out_shape=[{self.output_shape}]"


# MUST RENAME THIS LATER
@attrs.frozen(cache_hash=True)
class ConnectedConv2DTranspose(SingleInputLayer):
    input_layer: ConnectableLayer
    params: Conv2DTranspose

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.input_layer, ConnectableLayer)
        assert isinstance(self.params, Conv2DTranspose)

    @property
    def name(self) -> str:
        return self.params.name

    @property
    def output_shape(self) -> Shape:
        input_shape = self.input_layer.output_shape
        params = self.params
        out_width = input_shape.width * params.stride
        out_height = input_shape.height * params.stride
        out_depth = self.params.filter_count
        return Shape(width=out_width, height=out_height, depth=out_depth)

    def to_tensor(
        self,
        known_tensores: dict["ConnectableLayer", tf.Tensor],
    ) -> tf.Tensor:
        if self not in known_tensores:
            source = self.input_layer.to_tensor(known_tensores)

            params = self.params
            strides = (params.stride, params.stride)

            # layer = kl.Conv2DTranspose(
            #     filters=params.filter_count,
            #     kernel_size=params.kernel_size,
            #     strides=strides,
            #     padding="same",
            #     name=params.name,
            # )
            conv = kl.Conv2D(
                filters=params.filter_count,
                kernel_size=params.kernel_size,
                strides=1,
                padding="same",
                name=params.name + "_conv",
            )(source)
            upsample = kl.UpSampling2D(
                size=strides,
                name=params.name + "_upsample",
            )(conv)
            known_tensores[self] = upsample

        return known_tensores[self]

    def __repr__(self) -> str:
        return f"{self.params.name}, params={self.params}, input={self.input_layer}, out_shape=[{self.output_shape}]"


@attrs.frozen(cache_hash=True)
class ConnectedMaxPooling2D(SingleInputLayer):
    input_layer: ConnectableLayer
    params: MaxPool2D

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.input_layer, ConnectableLayer)
        assert isinstance(self.params, MaxPool2D)

    @property
    def name(self) -> str:
        return self.params.name

    @property
    def output_shape(self) -> Shape:
        input_shape = self.input_layer.output_shape
        params = self.params
        out_width = int(math.ceil(input_shape.width / params.stride))
        out_height = int(math.ceil(input_shape.height / params.stride))
        out_depth = input_shape.depth
        return Shape(width=out_width, height=out_height, depth=out_depth)

    def to_tensor(
        self,
        known_tensores: dict["ConnectableLayer", tf.Tensor],
    ) -> tf.Tensor:
        if self not in known_tensores:
            source = self.input_layer.to_tensor(known_tensores)

            params = self.params
            pool_size = (params.pool_size, params.pool_size)
            stride = (params.stride, params.stride)

            layer = kl.MaxPooling2D(
                pool_size=pool_size,
                strides=stride,
                padding="same",
                name=params.name,
            )

            tensor = layer(source)
            known_tensores[self] = tensor

        return known_tensores[self]

    def __repr__(self) -> str:
        return f"{self.params.name}: out_shape=[{self.output_shape}]"


@attrs.frozen(cache_hash=True)
class ConnectedAveragePooling2D(SingleInputLayer):
    input_layer: ConnectableLayer
    params: AvgPool2D

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.input_layer, ConnectableLayer)
        assert isinstance(self.params, AvgPool2D)

    @property
    def name(self) -> str:
        return self.params.name

    @property
    def output_shape(self) -> Shape:
        input_shape = self.input_layer.output_shape
        params = self.params
        out_width = int(math.ceil(input_shape.width / params.stride))
        out_height = int(math.ceil(input_shape.height / params.stride))
        out_depth = input_shape.depth
        return Shape(width=out_width, height=out_height, depth=out_depth)

    def to_tensor(
        self,
        known_tensores: dict["ConnectableLayer", tf.Tensor],
    ) -> tf.Tensor:
        if self not in known_tensores:
            source = self.input_layer.to_tensor(known_tensores)

            params = self.params
            pool_size = (params.pool_size, params.pool_size)
            stride = (params.stride, params.stride)

            layer = kl.AveragePooling2D(
                pool_size=pool_size,
                strides=stride,
                padding="same",
                name=params.name,
            )

            tensor = layer(source)
            known_tensores[self] = tensor

        return known_tensores[self]

    def __repr__(self) -> str:
        return f"{self.params.name}: out_shape=[{self.output_shape}]"


@attrs.frozen(cache_hash=True)
class ConnectedBatchNorm(SingleInputLayer):
    input_layer: ConnectableLayer
    params: BatchNorm

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.input_layer, ConnectableLayer)
        assert isinstance(self.params, BatchNorm)

    @property
    def name(self) -> str:
        return self.params.name

    @property
    def output_shape(self) -> Shape:
        return self.input_layer.output_shape

    def to_tensor(
        self,
        known_tensores: dict["ConnectableLayer", tf.Tensor],
    ) -> tf.Tensor:
        if self not in known_tensores:
            source = self.input_layer.to_tensor(known_tensores)
            layer = kl.BatchNormalization(name=self.params.name)
            tensor = layer(source)
            known_tensores[self] = tensor
        return known_tensores[self]

    def __repr__(self) -> str:
        return f"{self.params.name}: out_shape=[{self.output_shape}]"


@attrs.frozen(cache_hash=True)
class ConnectedAdd(MultiInputLayer):
    name: str
    input_layers: tuple[ConnectableLayer, ...]

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert self.name

        assert isinstance(self.input_layers, tuple)
        for layer in self.input_layers:
            assert isinstance(layer, ConnectableLayer)

        assert len(self.input_layers) > 1

        shapes = (layer.output_shape for layer in self.input_layers)
        if len(set(shapes)) != 1:
            raise ValueError("inputs must have the same shape")

        if len(set(self.input_layers)) != len(self.input_layers):
            raise ValueError("inputs must not contain duplicated layers")

    @property
    def output_shape(self) -> Shape:
        shape: Shape = self.input_layers[0].output_shape
        return shape

    def to_tensor(
        self,
        known_tensores: dict["ConnectableLayer", tf.Tensor],
    ) -> tf.Tensor:
        if self not in known_tensores:
            sources = [src.to_tensor(known_tensores) for src in self.input_layers]
            layer = kl.Add(name=self.name)
            tensor = layer(sources)
            known_tensores[self] = tensor
        return known_tensores[self]

    def __repr__(self) -> str:
        return f"{self.name}: out_shape=[{self.output_shape}]"


@attrs.frozen(cache_hash=True)
class ConnectedConcatenate(MultiInputLayer):
    name: str
    input_layers: tuple[ConnectableLayer, ...]

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.name, str)
        assert self.name

        assert isinstance(self.input_layers, tuple)
        for layer in self.input_layers:
            assert isinstance(layer, ConnectableLayer)

        assert len(self.input_layers) > 1

        for a, b in itertools.pairwise(self.input_layers):
            assert a.output_shape.width == b.output_shape.width
            assert a.output_shape.height == b.output_shape.height

        if len(set(self.input_layers)) != len(self.input_layers):
            raise ValueError("inputs must not contain duplicated layers")

    @property
    def output_shape(self) -> Shape:
        depths = (layer.output_shape.depth for layer in self.input_layers)
        total_depth = sum(depths)

        sample_shape = self.input_layers[0].output_shape
        return Shape(
            width=sample_shape.width,
            height=sample_shape.height,
            depth=total_depth,
        )

    def to_tensor(
        self,
        known_tensores: dict["ConnectableLayer", tf.Tensor],
    ) -> tf.Tensor:
        if self not in known_tensores:
            sources = [src.to_tensor(known_tensores) for src in self.input_layers]
            layer = kl.Concatenate(name=self.name)
            tensor = layer(sources)
            known_tensores[self] = tensor
        return known_tensores[self]

    def __repr__(self) -> str:
        return f"{self.name}: out_shape=[{self.output_shape}]"


@attrs.frozen(cache_hash=True)
class ConnectedRelu(SingleInputLayer):
    input_layer: ConnectableLayer
    params: Relu

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.input_layer, ConnectableLayer)
        assert isinstance(self.params, Relu)

    @property
    def name(self) -> str:
        return self.params.name

    @property
    def output_shape(self) -> Shape:
        return self.input_layer.output_shape

    def to_tensor(
        self,
        known_tensores: dict["ConnectableLayer", tf.Tensor],
    ) -> tf.Tensor:
        if self not in known_tensores:
            source = self.input_layer.to_tensor(known_tensores)
            layer = kl.Activation(
                activation=tf.nn.relu,
                name=self.params.name,
            )
            tensor = layer(source)
            known_tensores[self] = tensor
        return known_tensores[self]

    def __repr__(self) -> str:
        return f"{self.params.name}: out_shape=[{self.output_shape}]"


@attrs.frozen(cache_hash=True)
class ConnectedGelu(SingleInputLayer):
    input_layer: ConnectableLayer
    params: Gelu

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.input_layer, ConnectableLayer)
        assert isinstance(self.params, Gelu)

    @property
    def name(self) -> str:
        return self.params.name

    @property
    def output_shape(self) -> Shape:
        return self.input_layer.output_shape

    def to_tensor(
        self,
        known_tensores: dict["ConnectableLayer", tf.Tensor],
    ) -> tf.Tensor:
        if self not in known_tensores:
            source = self.input_layer.to_tensor(known_tensores)
            layer = kl.Activation(
                activation=tf.nn.gelu,
                name=self.params.name,
            )
            tensor = layer(source)
            known_tensores[self] = tensor
        return known_tensores[self]

    def __repr__(self) -> str:
        return f"{self.params.name}: out_shape=[{self.output_shape}]"


@attrs.frozen(cache_hash=True)
class ConnectedSwish(SingleInputLayer):
    input_layer: ConnectableLayer
    params: Swish

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.input_layer, ConnectableLayer)
        assert isinstance(self.params, Swish)

    @property
    def name(self) -> str:
        return self.params.name

    @property
    def output_shape(self) -> Shape:
        return self.input_layer.output_shape

    def to_tensor(
        self,
        known_tensores: dict["ConnectableLayer", tf.Tensor],
    ) -> tf.Tensor:
        if self not in known_tensores:
            source = self.input_layer.to_tensor(known_tensores)
            layer = kl.Activation(
                activation=tf.nn.swish,
                name=self.params.name,
            )
            tensor = layer(source)
            known_tensores[self] = tensor
        return known_tensores[self]

    def __repr__(self) -> str:
        return f"{self.params.name}: out_shape=[{self.output_shape}]"


def iter_sources(layer: ConnectableLayer) -> typing.Iterable[ConnectableLayer]:
    if isinstance(layer, MultiInputLayer):
        return layer.input_layers

    elif isinstance(layer, SingleInputLayer):
        return [layer.input_layer]

    else:
        return []
