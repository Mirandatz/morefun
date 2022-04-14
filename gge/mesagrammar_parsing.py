import pathlib
import typing

import lark
from loguru import logger

import gge.transformers

LOWER_GRAMMAR_PATH = (
    pathlib.Path(__file__).parent / "grammar_files" / "lower_grammar.lark"
)


def get_parser() -> lark.Lark:
    logger.debug("starting parsing mesagrammar")
    parser = lark.Lark.open(grammar_filename=str(LOWER_GRAMMAR_PATH), parser="lalr")
    logger.debug("finished parsing mesagrammar")
    return parser


def parse_mesagrammar_tokenstream(token_stream: str) -> lark.Tree[typing.Any]:
    logger.debug("starting parsing mesagrammar token stream")
    tree = get_parser().parse(token_stream)
    logger.debug("finished parsing mesagrammar token stream")
    return tree


@lark.v_args(inline=True)
class MesagrammarTransformer(gge.transformers.SinglePassTransformer):
    def INT(self, token: lark.Token) -> int:
        self._raise_if_not_running()
        return int(token.value)

    def FLOAT(self, token: lark.Token) -> float:
        self._raise_if_not_running()
        return float(token.value)

    def BOOL(self, token: lark.Token) -> bool:
        self._raise_if_not_running()

        if token.value == "true":
            return True

        elif token.value == "false":
            return False

        else:
            raise ValueError(f"unexpected `BOOL` value=<{token.value}>")
