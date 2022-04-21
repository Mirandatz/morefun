import typing

import attrs
import hypothesis.strategies as hs
import pytest
from hypothesis import given

import gge.composite_genotypes as cg
import gge.grammars as gr
import gge.layers as gl
import gge.neural_network as gnn
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


@hs.composite
def sgd_grammar(draw: hs.DrawFn) -> GrammarToPhenotypeTestData[optim.SGD]:
    learning_rate = draw(hs.floats(min_value=0, max_value=9, exclude_min=True))
    momentum = draw(hs.floats(min_value=0, max_value=1))
    nesterov = draw(hs.booleans())

    grammar = (
        'start : "sgd"'
        f'"learning_rate" {learning_rate}'
        f'"momentum" {momentum}'
        f'"nesterov" {str(nesterov).lower()}'
    )

    phenotype = optim.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
    )

    return GrammarToPhenotypeTestData(
        grammar=grammar,
        phenotype=phenotype,
    )


@hs.composite
def adam_grammar(draw: hs.DrawFn) -> GrammarToPhenotypeTestData[optim.Adam]:
    learning_rate = draw(hs.floats(min_value=0, max_value=9, exclude_min=True))
    beta1 = draw(
        hs.floats(min_value=0, max_value=1, exclude_min=True, exclude_max=True)
    )
    beta2 = draw(
        hs.floats(min_value=0, max_value=1, exclude_min=True, exclude_max=True)
    )
    epsilon = draw(
        hs.floats(min_value=0, max_value=1, exclude_min=True, exclude_max=True)
    )
    amsgrad = draw(hs.booleans())

    grammar = (
        'start : "adam"'
        f'"learning_rate" {learning_rate}'
        f'"beta1" {beta1}'
        f'"beta2" {beta2}'
        f'"epsilon" {epsilon}'
        f'"amsgrad" {str(amsgrad).lower()}'
    )

    phenotype = optim.Adam(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        amsgrad=amsgrad,
    )

    return GrammarToPhenotypeTestData(
        grammar=grammar,
        phenotype=phenotype,
    )


@hs.composite
def neural_network(draw: hs.DrawFn) -> GrammarToPhenotypeTestData[gnn.NeuralNetwork]:
    # TODO: Implement this.
    raise NotImplementedError()


@given(test_data=sgd_grammar())
def test_sgd(test_data: GrammarToPhenotypeTestData[optim.SGD]) -> None:
    """Can process a middle-grammar to generate a SGD optimizer."""
    grammar = gr.Grammar(test_data.grammar)
    genotype = sge.create_genotype(grammar, rng=rand.create_rng())
    tokenstream = sge.map_to_tokenstream(genotype, grammar)

    actual = optim.parse(tokenstream, start="optimizer")
    expected = test_data.phenotype
    assert expected == actual


@given(test_data=adam_grammar())
def test_adam(test_data: GrammarToPhenotypeTestData[optim.Adam]) -> None:
    """Can process a middle-grammar to generate an Adam optimizer."""
    grammar = gr.Grammar(test_data.grammar)
    genotype = sge.create_genotype(grammar, rng=rand.create_rng())
    tokenstream = sge.map_to_tokenstream(genotype, grammar)

    actual = optim.parse(tokenstream, start="optimizer")
    expected = test_data.phenotype
    assert expected == actual


@pytest.mark.xfail(reason="Not implemented")
@given(test_data=neural_network())
def test_network(test_data: GrammarToPhenotypeTestData[gnn.NeuralNetwork]) -> None:
    """Can process a middle-grammar to generate a NeuralNetwork."""
    grammar = gr.Grammar(test_data.grammar)
    composite_genotype = cg.create_genotype(grammar, rng=rand.create_rng())
    input_layer = gl.make_input(1, 1, 1)

    actual = gnn.make_network(composite_genotype, grammar, input_layer)
    expected = test_data.phenotype
    assert expected == actual
