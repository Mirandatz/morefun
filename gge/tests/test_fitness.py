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
    """Should select the individuals with largest fitnesses."""

    fittest = gfit.select_fittest(data.population, data.fittest_count)

    assert data.fittest_count == len(fittest)

    fitness_of_the_worst_selected_individual = min(fittest.values())
    for genotype, fitness in data.population.items():
        if fitness > fitness_of_the_worst_selected_individual:
            assert genotype in fittest
