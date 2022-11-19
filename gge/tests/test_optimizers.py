import datetime as dt

import hypothesis
from hypothesis import given

import gge.grammars.optimizers as optimizer_parsing
import gge.grammars.structured_grammatical_evolution as sge
import gge.grammars.upper_grammars as ugr
import gge.neural_networks.optimizers as optimizer
import gge.randomness as rand
import gge.tests.strategies.data_structures as ds
import gge.tests.strategies.optimizers as gos

# autouse fixtures
from gge.tests.fixtures import hide_gpu_from_tensorflow, remove_logger_sinks  # noqa


@given(test_data=gos.sgds().map(gos.sgd_grammar))
def test_sgds_middle_grammar(test_data: ds.ParsingTestData[optimizer.SGD]) -> None:
    """Can process a middle_grammar to generate a SGD optimizer."""

    grammar = ugr.Grammar(test_data.tokenstream)
    genotype = sge.create_genotype(grammar, rng=rand.create_rng())
    tokenstream = sge.map_to_tokenstream(genotype, grammar)
    phenotype = optimizer_parsing.parse(tokenstream, start="optimizer")

    assert test_data.parsed == phenotype


@given(sgd=gos.sgds())
def test_sgd_to_tensorflow(sgd: optimizer.SGD) -> None:
    """Can convert a SGD optimizer to its Tensorflow equivalent."""

    tf_sgd = sgd.to_tensorflow()

    assert tf_sgd.learning_rate == sgd.learning_rate
    assert tf_sgd.momentum == sgd.momentum
    assert tf_sgd.nesterov == sgd.nesterov


@given(test_data=gos.adams().map(gos.adam_grammar))
def test_adams_middle_grammar(
    test_data: ds.ParsingTestData[optimizer.Adam],
) -> None:
    """Can process a middle_grammar to generate an Adam optimizer."""

    grammar = ugr.Grammar(test_data.tokenstream)
    genotype = sge.create_genotype(grammar, rng=rand.create_rng())
    tokenstream = sge.map_to_tokenstream(genotype, grammar)
    phenotype = optimizer_parsing.parse(tokenstream, start="optimizer")

    assert test_data.parsed == phenotype


@given(adam=gos.adams())
def test_adam_to_tensorflow(adam: optimizer.Adam) -> None:
    """Can convert a Adam optimizer to its Tensorflow equivalent."""
    tf_adam = adam.to_tensorflow()

    assert tf_adam.learning_rate == adam.learning_rate
    assert tf_adam.beta_1 == adam.beta1
    assert tf_adam.beta_2 == adam.beta2
    assert tf_adam.epsilon == adam.epsilon
    assert tf_adam.amsgrad == adam.amsgrad


@given(ranger=gos.rangers())
def test_ranger_to_tensorflow(ranger: optimizer.Ranger) -> None:
    """Can convert a Adam optimizer to its Tensorflow equivalent."""
    tf_ranger = ranger.to_tensorflow()

    assert tf_ranger.learning_rate == ranger.learning_rate
    assert tf_ranger.sync_period == ranger.sync_period
    assert tf_ranger.slow_step_size == ranger.slow_step_size


@given(test_data=gos.rangers().map(gos.ranger_grammar))
@hypothesis.settings(deadline=dt.timedelta(seconds=2))
def test_rangers_middle_grammar(
    test_data: ds.ParsingTestData[optimizer.Adam],
) -> None:
    """Can process a middle_grammar to generate an Adam optimizer."""

    grammar = ugr.Grammar(test_data.tokenstream)
    genotype = sge.create_genotype(grammar, rng=rand.create_rng())
    tokenstream = sge.map_to_tokenstream(genotype, grammar)
    phenotype = optimizer_parsing.parse(tokenstream, start="optimizer")

    assert test_data.parsed == phenotype
