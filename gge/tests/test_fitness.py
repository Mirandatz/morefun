import numpy as np
import numpy.typing as npt
import pytest

import gge.evolutionary.fitnesses as gf


def test_fitness_evaluations_to_ndarray() -> None:
    me1 = gf.SuccessfulMetricEvaluation("a", 1, -1)
    me2 = gf.SuccessfulMetricEvaluation("b", 2, -2)
    fit1 = gf.Fitness((me1, me2))

    fitnesses = [fit1, fit1, fit1]
    actual = gf.fitnesses_to_ndarray(fitnesses)
    expected = np.asarray(
        [
            [me1.effective, me2.effective],
            [me1.effective, me2.effective],
            [me1.effective, me2.effective],
        ]
    )
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
