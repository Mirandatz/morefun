import typing

import attrs
import hypothesis.strategies as hs

import gge.grammars as gr  # noqa
import gge.name_generator
import gge.tests.strategies as ts


def gen_nonterminal_name(tp: type) -> str:
    name = gge.name_generator.NameGenerator().gen_name(tp)
    return name.lower()


@attrs.frozen
class Pool2DTestData:
    type: typing.Literal["max", "avg"]
    stride: int
    name: str

    def str_definition(self) -> str:
        return f'{self.name} : "pool2d" "{self.type}" "stride" {self.stride}'

    def nonterminal(self) -> gr.NonTerminal:
        return gr.NonTerminal(self.name)

    def expansion(self) -> gr.RuleOption:
        exp = ['"pool2d"', self.type, str(self.stride)]
        symbols = tuple(map(gr.Terminal, exp))
        return gr.RuleOption(symbols)


@hs.composite
def pool2ds(draw: ts.DrawStrat) -> Pool2DTestData:
    return Pool2DTestData(
        type=draw(hs.sampled_from(["max", "avg"])),
        stride=draw(hs.integers(min_value=1, max_value=9)),
        name=gen_nonterminal_name(Pool2DTestData),
    )


def make_raw_grammar(data: Pool2DTestData) -> str:
    return f"""start : {data.name}
               {data.str_definition()}"""
