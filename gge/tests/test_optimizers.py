from hypothesis import given

import gge.grammars as gr
import gge.optimizers as optim
import gge.randomness as rand
import gge.structured_grammatical_evolution as sge
import gge.tests.strategies.data_structures as ds
import gge.tests.strategies.optimizers as gos

# auto-import fixtures
from gge.tests.fixtures import hide_gpu_from_tensorflow, remove_logger_sinks  # noqa


@given(test_data=gos.sgds().map(gos.sgd_grammar))
def test_sgds_middle_grammar(test_data: ds.ParsingTestData[optim.SGD]) -> None:
    """Can process a middle_grammar to generate a SGD optimizer."""

    grammar = gr.Grammar(test_data.tokenstream)
    genotype = sge.create_genotype(grammar, rng=rand.create_rng())
    tokenstream = sge.map_to_tokenstream(genotype, grammar)
    phenotype = optim.parse(tokenstream, start="optimizer")

    assert test_data.parsed == phenotype


@given(sgd=gos.sgds())
def test_sgd_to_tensorflow(sgd: optim.SGD) -> None:
    """Can convert a SGD optimizer to its Tensorflow equivalent."""

    tf_sgd = sgd.to_tensorflow()

    assert tf_sgd.learning_rate == sgd.learning_rate
    assert tf_sgd.momentum == sgd.momentum
    assert tf_sgd.nesterov == sgd.nesterov


@given(test_data=gos.adams().map(gos.adam_grammar))
def test_adams_middle_grammar(test_data: ds.ParsingTestData[optim.Adam]) -> None:
    """Can process a middle_grammar to generate an Adam optimizer."""

    grammar = gr.Grammar(test_data.tokenstream)
    genotype = sge.create_genotype(grammar, rng=rand.create_rng())
    tokenstream = sge.map_to_tokenstream(genotype, grammar)
    phenotype = optim.parse(tokenstream, start="optimizer")

    assert test_data.parsed == phenotype


@given(adam=gos.adams())
def test_adam_to_tensorflow(adam: optim.Adam) -> None:
    """Can convert a Adam optimizer to its Tensorflow equivalent."""
    tf_adam = adam.to_tensorflow()

    assert tf_adam.learning_rate == adam.learning_rate
    assert tf_adam.beta_1 == adam.beta1
    assert tf_adam.beta_2 == adam.beta2
    assert tf_adam.epsilon == adam.epsilon
    assert tf_adam.amsgrad == adam.amsgrad
