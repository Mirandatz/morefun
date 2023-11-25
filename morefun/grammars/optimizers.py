import typing

import lark
import typeguard
from loguru import logger

import morefun.grammars.lower_grammar_parsing as lgp
import morefun.neural_networks.optimizers as go


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

    def optimizer(self, optimizer: go.Optimizer) -> go.Optimizer:
        self._raise_if_not_running()

        assert isinstance(optimizer, go.Optimizer)
        return optimizer

    @typeguard.typechecked
    def sgd(
        self,
        marker: None,
        learning_rate: float,
        momentum: float,
        nesterov: bool,
    ) -> go.SGD:
        self._raise_if_not_running()
        return go.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)

    @typeguard.typechecked
    def adam(
        self,
        marker: None,
        learning_rate: float,
        beta1: float,
        beta2: float,
        epsilon: float,
        amsgrad: bool,
    ) -> go.Adam:
        self._raise_if_not_running()
        return go.Adam(
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
    ) -> go.Ranger:
        self._raise_if_not_running()

        return go.Ranger(
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
) -> go.Optimizer:
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
    assert isinstance(optimizer, go.Optimizer)

    logger.debug("finished parsing optimizer tokenstream")
    return optimizer
