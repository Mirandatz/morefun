import typing

import attrs
import hypothesis.strategies as hs

T = typing.TypeVar("T", int, float, bool)


@attrs.frozen
class GrammarArgs(typing.Generic[T]):
    text: str
    values: tuple[T, ...]


@hs.composite
def parenthesized(draw: hs.DrawFn, arg: str) -> str:
    if draw(hs.booleans()):
        return f"({arg})"
    else:
        return arg


@hs.composite
def grammar_args(
    draw: hs.DrawFn,
    values_strategy: hs.SearchStrategy[list[T]],
) -> GrammarArgs[T]:
    values = draw(values_strategy)

    # .lower() is only useful/necessary if T==bool
    # but it doesn't change anythinf if T==int or T==float
    values_as_str = [str(v).lower() for v in values]

    if len(values) == 1:
        arg = values_as_str[0]
        text = draw(parenthesized(arg))
    else:
        text = f"({' | '.join(values_as_str)})"

    return GrammarArgs(text=text, values=tuple(values))
