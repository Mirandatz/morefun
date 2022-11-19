import abc
import typing

import attrs
import lark
import tensorflow as tf
import tensorflow_addons as tfa
import typeguard
from loguru import logger

import gge.grammars.lower_grammar_parsing as lgp


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


@typeguard.typechecked
@attrs.frozen
class Ranger(Optimizer):
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    amsgrad: bool
    sync_period: int
    slow_step_size: float

    def __attrs_post_init__(self) -> None:
        assert self.learning_rate > 0
        assert self.beta1 > 0
        assert self.beta2 > 0
        assert self.epsilon > 0
        assert self.sync_period >= 1
        assert self.slow_step_size > 0

    def to_tensorflow(self) -> tfa.optimizers.Lookahead:
        radam = tfa.optimizers.RectifiedAdam(
            learning_rate=self.learning_rate,
            beta_1=self.beta1,
            beta_2=self.beta2,
            epsilon=self.epsilon,
            amsgrad=self.amsgrad,
        )
        return tfa.optimizers.Lookahead(
            radam, sync_period=self.sync_period, slow_step_size=self.slow_step_size
        )


@lark.v_args(inline=True)
class OptimizerSynthetizer(lgp.LowerGrammarTransformer):

    # This set contains terminals that when visited/processed are just converted into `None`.
    # It is used in `OptimizerSynthetizer.__default_token__` to remove boilerplate code.
    _expected_tokens = {
        '"adam"',
        '"sgd"',
        '"ranger"',
        '"learning_rate"',
        '"momentum"',
        '"nesterov"',
        '"beta1"',
        '"beta2"',
        '"epsilon"',
        '"amsgrad"',
        '"sync_period"',
        '"slow_step_size"',
    }

    # This set contains the names of rules that when visited just return the value
    # of a "key value pair".
    _key_value_pair_rules = {
        "learning_rate",
        "momentum",
        "nesterov",
        "beta1",
        "beta2",
        "epsilon",
        "amsgrad",
        "sync_period",
        "slow_step_size",
    }

    def __default__(
        self,
        data: typing.Any,
        children: typing.Any,
        meta: typing.Any,
    ) -> typing.Any:
        if data.value in self._key_value_pair_rules:
            marker, value = children
            return value

        return super().__default__(data, children, meta)

    def __default_token__(self, token: lark.Token) -> typing.Any:
        self._raise_if_not_running()

        if token.value in self._expected_tokens:
            return None

        return super().__default_token__(token)

    def optimizer(self, optimizer: Optimizer) -> Optimizer:
        self._raise_if_not_running()

        assert isinstance(optimizer, Optimizer)
        return optimizer

    @typeguard.typechecked
    def sgd(
        self,
        marker: None,
        learning_rate: float,
        momentum: float,
        nesterov: bool,
    ) -> SGD:
        self._raise_if_not_running()
        return SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)

    @typeguard.typechecked
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
        return Adam(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            amsgrad=amsgrad,
        )

    @typeguard.typechecked
    def ranger(
        self,
        marker: None,
        learning_rate: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        amsgrad: bool,
        sync_period: int,
        slow_step_size: float,
    ) -> Ranger:
        self._raise_if_not_running()

        return Ranger(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            sync_period=sync_period,
            slow_step_size=slow_step_size,
        )


def parse(
    tokenstream: str,
    *,
    start: typing.Literal["start", "optimizer"] = "start",
) -> Optimizer:
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
