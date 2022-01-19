import datetime as dt
import pathlib
import typing

import gge.composite_genotypes as cg
import gge.fitnesses as fit
import gge.novelty as novel
import gge.population_mutation as gmut
import gge.randomness as rand

EvaluatedPopulation = dict[cg.CompositeGenotype, float]
Callback = typing.Callable[[EvaluatedPopulation], None]


def try_run_single_generation(
    population: EvaluatedPopulation,
    mut_params: gmut.PopulationMutationParameters,
    fit_params: fit.FitnessEvaluationParameters,
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
) -> EvaluatedPopulation | None:
    assert len(population) > 0

    old_genotypes = list(population.keys())

    mutants = gmut.try_mutate_population(
        old_genotypes,
        mut_params,
        rng,
        novelty_tracker,
    )

    if mutants is None:
        return None

    mutants_fitnesses = {m: fit.evaluate(m, fit_params) for m in mutants}
    next_gen_candidates = {**population, **mutants_fitnesses}
    fittest = fit.select_fittest(next_gen_candidates, fittest_count=len(population))

    return fittest


def run_evolutionary_loop(
    population: EvaluatedPopulation,
    max_generations: int,
    mut_params: gmut.PopulationMutationParameters,
    fit_params: fit.FitnessEvaluationParameters,
    callbacks: list[Callback],
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
) -> EvaluatedPopulation:
    assert len(population) > 0
    assert max_generations > 0

    for _ in range(max_generations):
        next_gen = try_run_single_generation(
            population,
            mut_params,
            fit_params,
            rng,
            novelty_tracker,
        )

        if next_gen is None:
            return population
        else:
            population = next_gen

        for cb in callbacks:
            cb(next_gen)

    return population


class Checkpoint:
    def __init__(
        self,
        output_dir: pathlib.Path,
        filenames_prefix: str | None = None,
    ) -> None:
        if not output_dir.exists():
            # log warning
            output_dir.mkdir(parents=True)

        if filenames_prefix is None:
            now = dt.datetime.now()
            filenames_prefix = (
                f"experiment_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}"
            )

        self._current_gen = 0
        self._output_dir = output_dir

        raise NotImplementedError()

    def __call__(self, population: EvaluatedPopulation) -> None:
        pass
