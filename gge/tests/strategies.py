import dataclasses
import typing

import hypothesis.strategies as hs


@dataclasses.dataclass(frozen=True)
class GrammarOption:
    """the mesagrammar '(1 | 2 | 3)' string has values [1, 2, 3]"""

    possible_values: list[int]

    @property
    def mesagrammar_string(self) -> str:
        return f"({'|'.join(map(str, self.possible_values))})"


@hs.composite
def grammar_integer_option(draw: typing.Callable[..., typing.Any]) -> GrammarOption:
    ints = draw(hs.lists(hs.integers(min_value=1), min_size=2, unique=True))
    return GrammarOption(ints)
