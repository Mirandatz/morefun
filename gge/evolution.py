import datetime as dt
import pathlib
import pickle
import sys
import typing

import numpy as np
from loguru import logger

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

        population = next_gen

        for cb in callbacks:
            cb(population)

    return population


class EntityNamer:
    def __init__(self) -> None:
        self._names: dict[typing.Any, str] = dict()

    def get_name(self, entity: typing.Any) -> str:
        if entity not in self._names:
            self._names[entity] = str(len(self._names))

        return self._names[entity]


def format_fitness(value: float) -> str:
    return format(value, "0.4f")


class Checkpoint:
    def __init__(
        self,
        output_dir: pathlib.Path,
        filenames_prefix: str | None = None,
    ) -> None:
        logger.trace("Checkpoint constructor")

        if not output_dir.exists():
            logger.info(
                f"Output dir does not exist and will be created at=<{output_dir}>"
            )
            output_dir.mkdir(parents=True)

        if filenames_prefix is None:
            now = dt.datetime.now()
            filenames_prefix = (
                f"experiment_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}"
            )

        self._current_gen = 0
        self._output_dir = output_dir
        self._prefix = filenames_prefix
        self._genotype_namer = EntityNamer()

    def __call__(self, population: EvaluatedPopulation) -> None:
        logger.trace("Checkpoint callback")

        assert len(population) > 0

        for genotype, fitness in population.items():
            filename = (
                self._prefix
                + f"_generation_{self._current_gen}"
                + f"_genotype_{self._genotype_namer.get_name(genotype)}"
                + f"_fitness_{format_fitness(fitness)}"
                + ".pickle"
            )

            path = self._output_dir / filename

            serialized = pickle.dumps(genotype)

            if path.is_file():
                old_data = path.read_bytes()
                if old_data != serialized:
                    logger.warning(
                        f"Genotype file already exist and will be overwritten; path=<{path}>"
                    )

            path.write_bytes(serialized)

        self._current_gen += 1


class PrintStatistics:
    def __init__(self) -> None:
        self._gen_nr = 0

    def __call__(self, population: EvaluatedPopulation) -> None:
        fitnesses = list(population.values())

        mean = np.mean(fitnesses)
        std = np.std(fitnesses)
        max_ = np.max(fitnesses)

        msg = (
            f"Finished generation=<{self._gen_nr}>,"
            + f" mean fit=>{format_fitness(mean)} +/- {format_fitness(std)}>,"
            + f" max fit=<{format_fitness(max_)}>"
        )

        logger.success(msg)

        self._gen_nr += 1
