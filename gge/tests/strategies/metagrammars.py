import attrs
import hypothesis.strategies as hs


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
def int_args(
    draw: hs.DrawFn,
    values_strategy: hs.SearchStrategy[list[int]],
) -> IntArgs:
    values = draw(values_strategy)
    values_as_str = [str(v) for v in values]

    if len(values) == 1:
        arg = values_as_str[0]
        text = draw(parenthesized_arg(arg))
    else:
        text = f"({' | '.join(values_as_str)})"

    return IntArgs(text=text, values=tuple(values))
