import attrs
import lark

import gge.transformers as gge_transformers


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
class OptimizerSynthetizer(gge_transformers.SinglePassTransformer):
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
