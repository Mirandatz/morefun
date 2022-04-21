import abc
import typing

import attrs
import lark
import tensorflow as tf
from loguru import logger

import gge.lower_grammar_parsing as lgp


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


@lark.v_args(inline=True)
class OptimizerSynthetizer(lgp.LowerGrammarTransformer):
    def optimizer(self, optimizer: Optimizer) -> Optimizer:
        self._raise_if_not_running()

        assert isinstance(optimizer, Optimizer)
        return optimizer

    def sgd(
        self,
        marker: None,
        learning_rate: float,
        momentum: float,
        nesterov: bool,
    ) -> SGD:
        self._raise_if_not_running()

        assert isinstance(learning_rate, float)
        assert isinstance(momentum, float)
        assert isinstance(nesterov, bool)

        return SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)

    def adam(
        self,
        marker: None,
        learning_rate: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        amsgrad: bool,
    ) -> Adam:
        self._raise_if_not_running()

        assert isinstance(learning_rate, float)
        assert isinstance(beta1, float)
        assert isinstance(beta2, float)
        assert isinstance(epsilon, float)
        assert isinstance(amsgrad, bool)

        return Adam(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            amsgrad=amsgrad,
        )

    def learning_rate(self, marker: None, value: float) -> float:
        self._raise_if_not_running()
        assert isinstance(value, float)
        return value

    def momentum(self, marker: None, value: float) -> float:
        self._raise_if_not_running()
        assert isinstance(value, float)
        return value

    def nesterov(self, marker: None, value: bool) -> bool:
        self._raise_if_not_running()
        assert isinstance(value, bool)
        return value

    def beta1(self, marker: None, value: float) -> float:
        self._raise_if_not_running()
        assert isinstance(value, float)
        return value

    def beta2(self, marker: None, value: float) -> float:
        self._raise_if_not_running()
        assert isinstance(value, float)
        return value

    def epsilon(self, arker: None, value: float) -> float:
        self._raise_if_not_running()
        assert isinstance(value, float)
        return value

    def amsgrad(self, marker: None, value: bool) -> bool:
        self._raise_if_not_running()
        assert isinstance(value, bool)
        return value

    def SGD(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def LEARNING_RATE(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def MOMENTUM(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def NESTEROV(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def ADAM(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def BETA1(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def BETA2(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def EPSILON(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def AMSGRAD(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None


def parse(tokenstream: str, start: typing.Literal["start", "optimizer"]) -> Optimizer:
    """
    `start` indicates whether `tokenstream`'s first symbol is
    the optimizer start symbol or the grammar start symbol.
    """

    logger.debug("parsing optimizer tokenstream")

    assert start in ("start", "optimizer")
    tree = lgp.parse_tokenstream(
        tokenstream,
        start=start,
        relevant_subtree="optimizer",
    )
    optimizer = OptimizerSynthetizer().transform(tree)
    assert isinstance(optimizer, Optimizer)

    logger.debug("finished parsing optimizer tokenstream")
    return optimizer
