import keras
import keras.layers as kl
import tensorflow as tf

import gge.layers as gl


def input_to_tensor(layer: gl.Input) -> tf.Tensor:
    shape = layer.shape
    return keras.Input(shape=(shape.width, shape.height, shape.depth))


def conv2d_to_tensor(layer: gl.ConnectedConv2D) -> tf.Tensor:
    source = layer_to_tensor(layer.input_layer)

    params = layer.params
    strides = (params.stride, params.stride)

    conv2d = kl.Conv2D(
        filters=params.filter_count,
        kernel_size=params.kernel_size,
        strides=strides,
        padding="same",
        name=params.name,
    )

    return conv2d(source)


def conv2dtranpose_to_tensor(layer: gl.ConnectedConv2DTranspose) -> tf.Tensor:
    source = layer_to_tensor(layer.input_layer)

    params = layer.params
    strides = (params.stride, params.stride)

    conv2d_trans = kl.Conv2DTranspose(
        filters=layer.params.filter_count,
        kernel_size=layer.params.kernel_size,
        strides=strides,
        padding="same",
        name=params.name,
    )

    return conv2d_trans(source)


def pool2d_to_tensor(layer: gl.ConnectedPool2D) -> tf.Tensor:
    source = layer_to_tensor(layer.input_layer)

    params = layer.params
    pooling_type = params.pooling_type
    pool_size = (params.stride, params.stride)

    match pooling_type:
        case gl.PoolType.MAX_POOLING:
            pool = kl.MaxPool2D(
                pool_size=pool_size,
                padding="same",
                name=params.name,
            )
            return pool(source)

        case gl.PoolType.AVG_POOLING:
            pool = kl.AveragePooling2D(
                pool_size=pool_size,
                padding="same",
                name=params.name,
            )
            return pool(source)

        case unknown:
            raise ValueError(f"unknown pooling type: {unknown}")


def batchnorm_to_tensor(layer: gl.ConnectedBatchNorm) -> tf.Tensor:
    source = layer_to_tensor(layer.input_layer)
    batchnorm = kl.BatchNormalization(name=layer.params.name)
    return batchnorm(source)


def add_to_tensor(layer: gl.ConnectedAdd) -> tf.Tensor:
    sources = [layer_to_tensor(src) for src in layer.input_layers]
    add = kl.Add(name=layer.name)
    return add(sources)


def concatenate_to_tensor(layer: gl.ConnectedConcatenate) -> tf.Tensor:
    sources = [layer_to_tensor(src) for src in layer.input_layers]
    concatenate = kl.Concatenate(name=layer.name)
    return concatenate(sources)


def relu_to_tensor(layer: gl.ConnectedRelu) -> tf.Tensor:
    source = layer_to_tensor(layer.input_layer)
    activation = kl.Activation(
        activation=tf.nn.relu,
        name=layer.params.name,
    )
    return activation(source)


def gelu_to_tensor(layer: gl.ConnectedGelu) -> tf.Tensor:
    source = layer_to_tensor(layer.input_layer)
    activation = kl.Activation(
        activation=tf.nn.gelu,
        name=layer.params.name,
    )
    return activation(source)


def swish_to_tensor(layer: gl.ConnectedSwish) -> tf.Tensor:
    source = layer_to_tensor(layer.input_layer)
    activation = kl.Activation(
        activation=tf.nn.swish,
        name=layer.params.name,
    )
    return activation(source)


def layer_to_tensor(layer: gl.ConnectableLayer) -> tf.Tensor:
    match layer:
        case gl.Input():
            return input_to_tensor(layer)

        case gl.ConnectedConv2D():
            conv2d_to_tensor(layer)

        case gl.ConnectedConv2DTranspose():
            conv2dtranpose_to_tensor(layer)

        case gl.ConnectedPool2D():
            return pool2d_to_tensor(layer)

        case gl.ConnectedBatchNorm():
            return batchnorm_to_tensor(layer)

        case gl.ConnectedAdd():
            return add_to_tensor(layer)

        case gl.ConnectedConcatenate():
            return concatenate_to_tensor(layer)

        case gl.ConnectedRelu():
            return relu_to_tensor(layer)

        case gl.ConnectedGelu():
            return gelu_to_tensor(layer)

        case gl.ConnectedSwish():
            return swish_to_tensor(layer)

        case unknown:
            raise ValueError(f"unknown layer type: {type(unknown)}")
