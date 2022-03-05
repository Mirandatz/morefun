import attrs
import lark

import gge.mesagrammar_parsing as mp


@attrs.frozen
class SGD:
    learning_rate: float
    momentum: float
    nesterov: bool

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.learning_rate, float)
        assert isinstance(self.momentum, float)
        assert isinstance(self.nesterov, bool)

        assert self.learning_rate > 0
        assert self.momentum >= 0


@lark.v_args(inline=True)
class OptimizerSynthetizer(mp.MesagrammarTransformer):
    def optimizer(self, optimizer: SGD) -> SGD:
        self._raise_if_not_running()

        assert isinstance(optimizer, SGD)
        return optimizer

    def sgd(
        self,
        learning_rate: float,
        momentum: float,
        nesterov: bool,
    ) -> SGD:
        self._raise_if_not_running()

        assert isinstance(learning_rate, float)
        assert isinstance(momentum, float)
        assert isinstance(nesterov, bool)

        return SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)

    def learning_rate(self, value: float) -> float:
        self._raise_if_not_running()
        assert isinstance(value, float)
        return value

    def momentum(self, value: float) -> float:
        self._raise_if_not_running()
        assert isinstance(value, float)
        return value

    def nesterov(self, value: bool) -> bool:
        self._raise_if_not_running()
        assert isinstance(value, bool)
        return value


def parse(token_stream: str) -> SGD:
    """
    This is not a "string deserialization function".
    The input string is expected to be a "token stream"
    that can be translated into an abstract syntax tree that can
    be visited/transformed into a `SGD`.
    """

    tree = mp.parse_mesagrammar_tokenstream(token_stream)
    relevant_subtrees = list(tree.find_data("optimizer"))
    assert len(relevant_subtrees) == 1

    optimizer_tree = relevant_subtrees[0]

    optimizer = OptimizerSynthetizer().transform(optimizer_tree)
    assert isinstance(optimizer, SGD)
    return optimizer
