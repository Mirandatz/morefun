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
class MaxPool2DTestData:
    pool_size: int
    stride: int
    name: str

    def str_definition(self) -> str:
        return f'{self.name} : "max_pool2d" "pool_size" {self.pool_size} "stride" {self.stride}'

    def nonterminal(self) -> gr.NonTerminal:
        return gr.NonTerminal(self.name)

    def expansion(self) -> gr.RuleOption:
        exp_texts = [
            '"max_pool2d"',
            '"pool_size"',
            str(self.pool_size),
            '"stride"',
            str(self.stride),
        ]
        symbols = tuple(map(gr.Terminal, exp_texts))
        return gr.RuleOption(symbols)


@hs.composite
def max_pool2ds(draw: ts.DrawStrat) -> MaxPool2DTestData:
    return MaxPool2DTestData(
        pool_size=draw(hs.integers(min_value=1, max_value=9)),
        stride=draw(hs.integers(min_value=1, max_value=9)),
        name=gen_nonterminal_name(MaxPool2DTestData),
    )


def make_raw_grammar(layers: list[MaxPool2DTestData] | None = None) -> str:
    if layer is None:
        raise NotImplementedError()

    layer_names = [layer.name for layer in layers]

    start_rule = "start" + " ".join(layer_names)

    return f"""start : {data.name}
               {data.str_definition()}"""
