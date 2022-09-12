import datetime as dt

import numpy as np
import numpy.typing as npt
import pytest

import gge.composite_genotypes as gc
import gge.fitnesses as gf
import gge.grammars as gr
import gge.randomness as rand
import gge.structured_grammatical_evolution as sge


def test_fitness_evaluations_to_ndarray() -> None:
    grammar = gr.Grammar(
        """
        start     : convblock optim
        convblock : conv~2
        conv      : "conv" "filter_count" "1" "kernel_size" "2" "stride" "3"
        optim     : "sgd" "learning_rate" "0.001" "momentum" "0.1" "nesterov" "false"
    """
    )
    rng = rand.create_rng(seed=0)
    backbone_genotype = sge.create_genotype(grammar, rng)
    composite_genotype = gc.make_composite_genotype(
        backbone_genotype,
        grammar,
        rng,
    )

    objective_value = 123

    fitness = gf.Fitness(
        names=tuple(["dummy_metric"]),
        values=tuple([objective_value]),
    )

    fer = gf.SuccessfulEvaluationResult(
        composite_genotype, fitness, dt.datetime.now(), dt.datetime.now()
    )

    fitnesses = [fer, fer, fer]
    actual = gf.fitness_evaluations_to_ndarray(fitnesses)
    expected = np.asarray([[objective_value], [objective_value], [objective_value]])
    assert np.array_equal(expected, actual)


@pytest.mark.parametrize(
    "data, fittest_count, expected_result",
    [
        (
            np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]),
            1,
            [3],
        ),
        (
            np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]),
            2,
            [3, 1],
        ),
        (
            np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]),
            3,
            [3, 1, 2],
        ),
        (
            np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]]),
            4,
            [3, 1, 2, 0],
        ),
        (
            np.asarray([[0], [1], [2], [3], [4]]),
            1,
            [4],
        ),
    ],
)
def test_nsga2(
    data: npt.NDArray[np.float64],
    fittest_count: int,
    expected_result: list[int],
) -> None:
    actual = gf.argsort_nsga2(data, fittest_count)
    assert expected_result == actual
