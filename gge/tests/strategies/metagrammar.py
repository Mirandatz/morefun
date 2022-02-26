import itertools
import typing

import attrs
import hypothesis.strategies as hs

import gge.grammars as gr
import gge.name_generator as namegen
import gge.tests.strategies.original_strategies as gge_os


@attrs.frozen
class IntArgs:
    text: str
    values: tuple[int, ...]


@hs.composite
def parenthesized_arg(draw: hs.DrawFn, arg: str) -> str:
    if draw(hs.booleans()):
        return f"({arg})"
    else:
        return arg


@hs.composite
def int_args(draw: hs.DrawFn, values: tuple[int, ...]) -> IntArgs:
    values_as_str = [str(v) for v in values]

    if len(values) == 1:
        arg = values_as_str[0]
        text = draw(parenthesized_arg(arg))
    else:
        text = f"({' | '.join(values_as_str)})"

    return IntArgs(text=text, values=values)


@attrs.frozen
class GrammarLine:
    text: str
    parsed: tuple[gr.ProductionRule, ...]


@attrs.frozen
class GrammarTestData:
    lines: tuple[GrammarLine, ...]


@hs.composite
def maxpool2d_def(draw: hs.DrawFn) -> GrammarLine:
    pool_sizes = draw(
        hs.lists(
            elements=hs.integers(min_value=1, max_value=9),
            min_size=1,
            max_size=3,
        )
    )
    strides = draw(
        hs.lists(
            elements=hs.integers(min_value=1, max_value=9),
            min_size=1,
            max_size=3,
        )
    )
    raise NotImplementedError()
