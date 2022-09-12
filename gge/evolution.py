import pathlib

from loguru import logger

import gge.fitnesses as gf
import gge.mutations as gm
import gge.novelty as novel
import gge.persistence
import gge.randomness as rand


def run_single_generation(
    population: list[gf.FitnessEvaluationResult],
    mut_params: gm.PopulationMutationParameters,
    fit_params: gf.FitnessEvaluationParameters,
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
) -> list[gf.FitnessEvaluationResult] | None:
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

    evaluated_mutants = [gf.evaluate(m, fit_params) for m in mutants]
    next_gen_candidates = population + evaluated_mutants
    fittest = gf.select_fittest_nsga2(next_gen_candidates, len(population))

    return fittest


def run_evolutionary_loop(
    starting_generation_number: int,
    number_of_generations_to_run: int,
    initial_population: list[gf.FitnessEvaluationResult],
    mutation_params: gm.PopulationMutationParameters,
    fitness_params: gf.FitnessEvaluationParameters,
    novelty_tracker: novel.NoveltyTracker,
    rng: rand.RNG,
    output_dir: pathlib.Path,
) -> None:
    assert len(initial_population) > 0
    assert number_of_generations_to_run > 0

    output_dir.mkdir(parents=True, exist_ok=True)

    population = list(initial_population)

    for gen_nr in range(
        starting_generation_number,
        starting_generation_number + number_of_generations_to_run,
    ):
        logger.info(f"started running generation {gen_nr}")

        maybe_population = run_single_generation(
            population=population,
            mut_params=mutation_params,
            fit_params=fitness_params,
            rng=rng,
            novelty_tracker=novelty_tracker,
        )

        if maybe_population is None:
            logger.info(
                "stopping evolutionary loop: reason=<unable to generate enough novel mutants>"
            )
            break
        else:
            population = maybe_population

        gge.persistence.save_generational_artifacts(
            generation_number=gen_nr,
            fittest=population,
            rng=rng,
            novelty_tracker=novelty_tracker,
            output_dir=output_dir,
        )

        logger.info(f"finished running generation {gen_nr}")
