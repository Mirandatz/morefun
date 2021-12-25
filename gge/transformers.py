import enum
import typing

import lark

_T = typing.TypeVar("_T")


class State(enum.Enum):
    READY_TO_RUN = enum.auto()
    RUNNING = enum.auto()
    DONE_RUNNING = enum.auto()


class DisposableTransformer(lark.Transformer[_T]):
    def __init__(self) -> None:
        self._state = State.READY_TO_RUN
        super().__init__(visit_tokens=True)

    def __default__(
        self,
        data: typing.Any,
        children: typing.Any,
        meta: typing.Any,
    ) -> None:
        raise NotImplementedError(f"method not implemented for tree.data: {data}")

    def __default_token__(self, token_text: typing.Any) -> None:
        raise NotImplementedError(
            f"method not implemented for token with text: {token_text}"
        )

    def transform(self, tree: lark.Tree) -> _T:
        assert self._state == State.READY_TO_RUN

        self._state = State.RUNNING

        result = super().transform(tree)

        self._state = State.DONE_RUNNING

        return result

    def _raise_if_not_running(self) -> None:
        if self._state != State.RUNNING:
            raise ValueError(
                "instances of this class should only be used once "
                "and only the method `transform` should be called manually."
            )

    def _raise_if_not_done(self) -> None:
        if self._state != State.DONE_RUNNING:
            raise ValueError(
                "this method should only be called after `transform` has been called"
            )
