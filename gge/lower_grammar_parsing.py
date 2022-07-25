import pathlib
import typing

import lark
from loguru import logger

import gge.transformers

LOWER_GRAMMAR_PATH = (
    pathlib.Path(__file__).parent / "grammar_files" / "lower_grammar.lark"
)


def get_parser(start: str) -> lark.Lark:
    logger.trace("starting parsing lower grammar")
    parser = lark.Lark.open(
        grammar_filename=str(LOWER_GRAMMAR_PATH),
        start=start,
        parser="lalr",
    )
    logger.trace("finished parsing lower grammar")
    return parser


def parse_tokenstream(
    tokenstream: str,
    start: str,
    relevant_subtree: str,
) -> lark.Tree[typing.Any]:
    """
    Parses `tokenstream` using the lower_grammar,
    assuming the "start symbol" of the grammar is `start` generating
    a tree.
    Returns the subtree associated of the `relevant_subtree` node.
    """

    logger.trace("starting parsing lower grammar tokenstream")
    tree = get_parser(start).parse(tokenstream)
    subtrees = list(tree.find_data(relevant_subtree))
    assert len(subtrees) == 1

    logger.trace("finished parsing lower grammar tokenstream")
    return subtrees[0]


@lark.v_args(inline=True)
class LowerGrammarTransformer(gge.transformers.SinglePassTransformer):
    def QUOTED_INT(self, token: lark.Token) -> int:
        self._raise_if_not_running()

        quoted_int = token.value
        assert quoted_int[0] == quoted_int[-1] == '"'

        unquoted = token[1:-1]
        return int(unquoted)

    def QUOTED_FLOAT(self, token: lark.Token) -> float:
        self._raise_if_not_running()

        quoted_float = token.value
        assert quoted_float[0] == quoted_float[-1] == '"'

        unquoted = token[1:-1]
        return float(unquoted)

    def QUOTE(self, token: lark.Token) -> None:
        self._raise_if_not_running()
        return None

    def BOOL(self, token: lark.Token) -> bool:
        self._raise_if_not_running()

        if token.value == '"true"':
            return True

        elif token.value == '"false"':
            return False

        else:
            raise ValueError(f"unexpected `BOOL` value=<{token.value}>")
