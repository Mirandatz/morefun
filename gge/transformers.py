import enum
import typing

from lark.tree import Tree as LarkTree
from lark.visitors import Transformer as LarkTransformer

_T = typing.TypeVar("_T")


class TransformerState(enum.Enum):
    READY = enum.auto()
    PARSING = enum.auto()
    PARSE_DONE = enum.auto()


class SinglePassTransformer(LarkTransformer[typing.Any, _T]):
    """
    Specialization of a base transformer that:
    - can only visit a tree if they start on the root node;
    - raises exceptions if they visit an unexpected subtree/token;
    - can only visit one tree.

    The last property makes instances of this class effectively "disposable"
    because they should be used once and then discarded. This "disposableness"
    reduces the complexity of the object by not keeping track / resetting
    internal states between calls to `transform`.
    """

    def __init__(self) -> None:
        self._state = TransformerState.READY
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

    def transform(self, tree: LarkTree[_T]) -> _T:
        assert self._state == TransformerState.READY

        self._state = TransformerState.PARSING

        result = super().transform(tree)

        self._state = TransformerState.PARSE_DONE

        return result

    def _raise_if_not_running(self) -> None:
        if self._state != TransformerState.PARSING:
            raise ValueError(
                "instances of this class should only be used once "
                "and only the method `transform` should be called manually."
            )

    def _raise_if_not_done(self) -> None:
        if self._state != TransformerState.PARSE_DONE:
            raise ValueError(
                "this method should only be called after `transform` has been called"
            )
