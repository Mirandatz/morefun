import dataclasses
import typing

import hypothesis.strategies as hs


@dataclasses.dataclass
class GrammarOption:
    """the mesagrammar '(1 | 2 | 3)' string has values [1, 2, 3]"""

    possible_values: list[int]
    mesagrammar_string: str


@hs.composite
def grammar_integer_option(draw: typing.Callable[..., typing.Any]) -> GrammarOption:
    ints = draw(hs.lists(hs.integers(min_value=1), min_size=2, unique=True))
    composite_string = f"({'|'.join(map(str, ints))})"
    return GrammarOption(ints, composite_string)
