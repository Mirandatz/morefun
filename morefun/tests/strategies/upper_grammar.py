import hypothesis.strategies as hs

import morefun.grammars.upper_grammars as ugr
import morefun.tests.strategies.data_structures

GrammarArgs = morefun.tests.strategies.data_structures.ParsingTestData[
    tuple[ugr.Terminal, ...]
]


@hs.composite
def maybe_parenthesize(draw: hs.DrawFn, arg: str) -> str:
    if draw(hs.booleans()):
        return f"({arg})"
    else:
        return arg


@hs.composite
def grammar_text(draw: hs.DrawFn, tokens: list[str]) -> str:
    if len(tokens) == 1:
        return draw(maybe_parenthesize(tokens[0]))
    else:
        return f"({' | '.join(tokens)})"


@hs.composite
def bool_args(draw: hs.DrawFn) -> GrammarArgs:
    values = draw(
        hs.lists(
            elements=hs.booleans(),
            min_size=1,
            max_size=3,
        )
    )

    tokens = ['"true"' if v else '"false"' for v in values]
    terminals = tuple([ugr.Terminal(v) for v in tokens])
    text = draw(grammar_text(tokens))

    return GrammarArgs(text, terminals)


@hs.composite
def flip_modes(draw: hs.DrawFn) -> GrammarArgs:
    tokens = draw(
        hs.lists(
            elements=hs.sampled_from(
                ['"horizontal"', '"vertical"', '"horizontal_and_vertical"']
            ),
            min_size=1,
            max_size=3,
        )
    )
    terminals = tuple([ugr.Terminal(t) for t in tokens])
    tokenstream = draw(grammar_text(tokens))
    return GrammarArgs(tokenstream, terminals)


@hs.composite
def int_args(draw: hs.DrawFn, min_value: int, max_value: int) -> GrammarArgs:
    values = draw(
        hs.lists(
            elements=hs.integers(
                min_value=min_value,
                max_value=max_value,
            ),
            min_size=1,
            max_size=3,
        )
    )

    tokens = [f'"{v}"' for v in values]
    terminals = tuple([ugr.Terminal(v) for v in tokens])
    text = draw(grammar_text(tokens))

    return GrammarArgs(text, terminals)


@hs.composite
def float_args(
    draw: hs.DrawFn,
    min_value: float | None = None,
    max_value: float | None = None,
    exclude_min: bool = False,
    exclude_max: bool = False,
) -> GrammarArgs:
    values = draw(
        hs.lists(
            elements=hs.floats(
                min_value=min_value,
                max_value=max_value,
                exclude_min=exclude_min,
                exclude_max=exclude_max,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=3,
        )
    )

    tokens = [f'"{v}"' for v in values]
    terminals = tuple([ugr.Terminal(v) for v in tokens])
    text = draw(grammar_text(tokens))

    return GrammarArgs(text, terminals)
