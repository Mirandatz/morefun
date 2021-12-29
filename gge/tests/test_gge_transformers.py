import lark
import pytest

import gge.transformers as tr


class BasicDisposableTransformer(tr.SinglePlassTransformer[None]):
    def __init__(self) -> None:
        super().__init__()

    def start(self, tree: lark.Tree) -> None:
        return None

    def EXAMPLE_TERMINAL(self, token: lark.Token) -> None:
        return None


def sample_grammar() -> str:
    return """
    start : example_nonterminal
          | EXAMPLE_TERMINAL
          | /"regex"/

    example_nonterminal : EXAMPLE_TERMINAL~2
    EXAMPLE_TERMINAL    : "ew"

    %import common.CNAME -> NAME
    %import common.WS

    %ignore WS
    """


def get_parser() -> lark.Lark:
    return lark.Lark(
        grammar=sample_grammar(),
        parser="lalr",
        maybe_placeholders=True,
    )


def extract_ast(text: str) -> lark.Tree:
    parser = get_parser()
    return parser.parse(text)


def test_raise_on_default_tree() -> None:
    text = """
        ew ew
    """
    tree = extract_ast(text)
    trans = BasicDisposableTransformer()
    with pytest.raises(
        NotImplementedError,
        match="method not implemented for tree.data: example_nonterminal",
    ):
        trans.transform(tree)


def test_raise_on_default_token() -> None:
    text = """
        ew
    """
    tree = extract_ast(text)
    trans = tr.SinglePlassTransformer()  # type: ignore

    with pytest.raises(
        NotImplementedError,
        match="method not implemented for token with text: ew",
    ):
        trans.transform(tree)
