import pathlib

from loguru import logger

import gge.fitnesses as cf
import gge.novelty as novel
import gge.persistence
import gge.population_mutation as gm
import gge.randomness as rand


def run_single_generation(
    population: list[cf.FitnessEvaluationResult],
    mut_params: gm.PopulationMutationParameters,
    fit_params: cf.FitnessEvaluationParameters,
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
) -> list[cf.FitnessEvaluationResult] | None:
    """
    Runs a single generation of the evolutionary loop and
    returns the fittest individuals genereated.

    If it is not possible to complete the generation process,
    (e.g., because the mutation procedure was unable to
    generate novelty individuals), this function returns
    `None` instead.
    """

    assert len(population) > 0

    initial_genotypes = [res.genotype for res in population]

    mutants = gm.try_mutate_population(
        initial_genotypes,
        mut_params,
        rng,
        novelty_tracker,
    )

    if mutants is None:
        return None

    evaluated_mutants = [cf.evaluate(m, fit_params) for m in mutants]
    next_gen_candidates = population + evaluated_mutants

    fittest = cf.select_fittest(
        next_gen_candidates,
        metric=cf.get_effective_fitness,
        fittest_count=len(population),
    )

    return fittest


def run_evolutionary_loop(
    starting_generation_number: int,
    number_of_generations_to_run: int,
    initial_population: list[cf.FitnessEvaluationResult],
    mutation_params: gm.PopulationMutationParameters,
    fitness_params: cf.FitnessEvaluationParameters,
    novelty_tracker: novel.NoveltyTracker,
    rng: rand.RNG,
    output_directory: pathlib.Path,
) -> None:
    assert len(initial_population) > 0
    assert number_of_generations_to_run > 0

    output_directory.mkdir(parents=True, exist_ok=True)

    population = list(initial_population)

    for gen_nr in range(
        starting_generation_number,
        starting_generation_number + number_of_generations_to_run,
    ):
        logger.info(f"started running generation {gen_nr}")

        population = run_single_generation(
            population,
            mutation_params,
            fitness_params,
            novelty_tracker=novelty_tracker,
            rng=rng,
        )

        if population is None:
            logger.info(
                "stopping evolutionary loop: reason=<unable to generate enough novel mutants>"
            )
            break

        gge.persistence.save_generation_output(
            generation_number=gen_nr,
            fittest=population,
            rng=rng,
            novelty_tracker=novelty_tracker,
        )

        logger.info(f"finished running generation {gen_nr}")
