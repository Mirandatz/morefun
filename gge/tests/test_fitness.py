import attrs
import hypothesis.strategies as hs
from hypothesis import given

import gge.fitnesses as gf


@attrs.frozen
class FitnessTestData:
    fitnesses: dict[int, float]
    fittest_count: int


@hs.composite
def fitness_test_data(draw: hs.DrawFn) -> FitnessTestData:
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

    fittest = gf.select_fittest(
        candidates=data.fitnesses.keys(),
        metric=lambda k: data.fitnesses[k],
        fittest_count=data.fittest_count,
    )

    assert data.fittest_count == len(fittest)

    worst_selected_individual = min(
        fittest,
        key=lambda k: data.fitnesses[k],
    )
    fitness_of_the_worst_selected_individual = data.fitnesses[worst_selected_individual]

    for genotype, fitness in data.fitnesses.items():
        if fitness > fitness_of_the_worst_selected_individual:
            assert genotype in fittest
