import abc

import attrs
import lark

import gge.lower_gramamar_parsing as lgp


class Optimizer(abc.ABC):
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
        assert self.momentum >= 0


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


@lark.v_args(inline=True)
class OptimizerSynthetizer(lgp.MesagrammarTransformer):
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

    def beta1(self, value: float) -> float:
        self._raise_if_not_running()
        assert isinstance(value, float)
        return value

    def beta2(self, value: float) -> float:
        self._raise_if_not_running()
        assert isinstance(value, float)
        return value

    def epsilon(self, value: float) -> float:
        self._raise_if_not_running()
        assert isinstance(value, float)
        return value

    def amsgrad(self, value: bool) -> bool:
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


def parse(token_stream: str) -> Optimizer:
    """
    This is not a "string deserialization function".
    The input string is expected to be a "token stream"
    that can be translated into an abstract syntax tree that can
    be visited/transformed into a `SGD`.
    """

    tree = lgp.parse_mesagrammar_tokenstream(token_stream)
    relevant_subtrees = list(tree.find_data("optimizer"))
    assert len(relevant_subtrees) == 1

    optimizer_tree = relevant_subtrees[0]

    optimizer = OptimizerSynthetizer().transform(optimizer_tree)
    assert isinstance(optimizer, Optimizer)
    return optimizer
