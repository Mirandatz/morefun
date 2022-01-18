import dataclasses
import typing

import hypothesis.strategies as hs
from hypothesis import given

import gge.fitnesses as gfit

DrawStrat = typing.Callable[..., typing.Any]


@dataclasses.dataclass(frozen=True)
class FitnessTestData:
    population: dict[int, float]
    fittest_count: int


@hs.composite
def fitness_test_data(draw: DrawStrat) -> FitnessTestData:
    population = draw(
        hs.dictionaries(
            keys=hs.integers(),
            values=hs.floats(allow_nan=False),
            min_size=1,
        )
    )

    fittest_count = draw(hs.integers(min_value=1, max_value=len(population)))

    return FitnessTestData(population, fittest_count)


@given(fitness_test_data())
def test_select_fittest(data: FitnessTestData) -> None:
    """Select fittest."""

    fittest = gfit.select_fittest(data.population, data.fittest_count)

    assert data.fittest_count == len(fittest)

    worst = min(fittest.values())
    for k, v in data.population.items():
        if v > worst:
            assert k in fittest
