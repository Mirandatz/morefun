import abc

import attrs
import tensorflow as tf


class Optimizer(abc.ABC):
    ...

    def to_tensorflow(self) -> tf.keras.optimizers.Optimizer:
        ...


@attrs.frozen
class SGD(Optimizer):
    learning_rate: float
    momentum: float
    nesterov: bool

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.learning_rate, float)
        assert isinstance(self.momentum, float)
        assert isinstance(self.nesterov, bool)

        assert self.learning_rate > 0
        assert 0 <= self.momentum <= 1

    def to_tensorflow(self) -> tf.keras.optimizers.SGD:
        return tf.keras.optimizers.SGD(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )


@attrs.frozen
class Adam(Optimizer):
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    amsgrad: bool

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.learning_rate, float)
        assert isinstance(self.beta1, float)
        assert isinstance(self.beta2, float)
        assert isinstance(self.epsilon, float)
        assert isinstance(self.amsgrad, bool)

        assert self.learning_rate > 0
        assert self.beta1 > 0
        assert self.beta2 > 0
        assert self.epsilon > 0

    def to_tensorflow(self) -> tf.keras.optimizers.Adam:
        return tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta1,
            beta_2=self.beta2,
            epsilon=self.epsilon,
            amsgrad=self.amsgrad,
        )
