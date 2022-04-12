import typing

import attrs
import hypothesis.strategies as hs
from hypothesis import given

import gge.grammars as gr
import gge.optimizers as optim
import gge.randomness as rand
import gge.structured_grammatical_evolution as sge

T = typing.TypeVar("T")


@attrs.frozen
class GrammarToPhenotypeTestData(typing.Generic[T]):
    grammar: str
    phenotype: T


@attrs.frozen
class DummyLayerDef:
    symbol: str
    rule: str


DUMMY_LAYER = DummyLayerDef(
    symbol="dummy",
    rule='dummy: "batchnorm"',
)


@hs.composite
def sgd_grammar(draw: hs.DrawFn) -> GrammarToPhenotypeTestData[optim.SGD]:
    learning_rate = draw(hs.floats(min_value=0, max_value=9, exclude_min=True))
    momentum = draw(hs.floats(min_value=0, max_value=9))
    nesterov = draw(hs.booleans())

    # must inject a dummy layer data to enable parsing
    grammar = f"""
    start : {DUMMY_LAYER.symbol} relevant_symbol
    {DUMMY_LAYER.rule}
    relevant_symbol : "sgd" "learning_rate" {learning_rate} "momentum" {momentum} "nesterov" {str(nesterov).lower()}
    """

    phenotype = optim.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
    )

    return GrammarToPhenotypeTestData(
        grammar=grammar,
        phenotype=phenotype,
    )


@given(test_data=sgd_grammar())
def test_sgd(test_data: GrammarToPhenotypeTestData[optim.SGD]) -> None:
    """Can process a middle-grammar to generate a SGD optimizer."""
    grammar = gr.Grammar(test_data.grammar)
    genotype = sge.create_genotype(grammar, rng=rand.create_rng())
    tokenstream = sge.map_to_tokenstream(genotype, grammar)

    actual = optim.parse(tokenstream)
    expected = test_data.phenotype
    assert expected == actual
