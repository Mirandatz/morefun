import pathlib
import pickle

from loguru import logger

import gge.composite_genotypes as cg
import gge.fitnesses as fit
import gge.novelty as novel
import gge.population_mutation as gmut
import gge.randomness as rand


def save_genotypes(
    genotypes: list[cg.CompositeGenotype], output_dir: pathlib.Path
) -> None:
    assert output_dir.is_dir()
    assert len(genotypes) > 0

    for geno in genotypes:
        filename = output_dir / f"{geno.unique_id}.genotype"
        serialized = pickle.dumps(geno, protocol=pickle.HIGHEST_PROTOCOL)
        filename.write_bytes(serialized)


def save_evaluation_results(
    results: list[fit.FitnessEvaluationResult], output_dir: pathlib.Path
) -> None:
    assert output_dir.is_dir()
    assert len(results) > 0

    for res in results:
        filename = output_dir / f"{res.genotype.unique_id}.evaluation_result"
        serialized = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
        filename.write_bytes(serialized)


def try_run_single_generation(
    population: list[fit.FitnessEvaluationResult],
    mut_params: gmut.PopulationMutationParameters,
    fit_params: fit.FitnessEvaluationParameters,
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
    output_dir: pathlib.Path,
) -> list[fit.FitnessEvaluationResult] | None:
    assert len(population) > 0
    assert output_dir.is_dir()

    initial_genotypes = [res.genotype for res in population]

    initial_genotypes_dir = output_dir / "initial_genotypes"
    initial_genotypes_dir.mkdir()
    save_genotypes(initial_genotypes, initial_genotypes_dir)

    mutants = gmut.try_mutate_population(
        initial_genotypes,
        mut_params,
        rng,
        novelty_tracker,
    )

    if mutants is None:
        return None

    mutants_dir = output_dir / "mutants"
    mutants_dir.mkdir()
    save_genotypes(mutants, mutants_dir)

    evaluated_mutants = [fit.evaluate(m, fit_params) for m in mutants]
    next_gen_candidates = population + evaluated_mutants

    next_gen_candidates_dir = output_dir / "next_gen_candidates"
    next_gen_candidates_dir.mkdir()
    save_evaluation_results(next_gen_candidates, next_gen_candidates_dir)

    fittest = fit.select_fittest(
        next_gen_candidates,
        metric=fit.get_effective_fitness,
        fittest_count=len(population),
    )

    fittest_dir = output_dir / "fittest"
    fittest_dir.mkdir()
    save_evaluation_results(fittest, fittest_dir)

    return fittest


def run_evolutionary_loop(
    population: list[fit.FitnessEvaluationResult],
    max_generations: int,
    mut_params: gmut.PopulationMutationParameters,
    fit_params: fit.FitnessEvaluationParameters,
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
    output_dir: pathlib.Path,
) -> list[fit.FitnessEvaluationResult]:
    assert len(population) > 0
    assert max_generations > 0
    assert output_dir.is_dir()

    for gen_nr in range(max_generations):
        logger.info(f"started generation {gen_nr}")

        generation_output_dir = output_dir / f"generation_{gen_nr}"
        generation_output_dir.mkdir()

        next_gen = try_run_single_generation(
            population,
            mut_params,
            fit_params,
            rng,
            novelty_tracker,
            generation_output_dir,
        )

        if next_gen is None:
            logger.info(
                "stopping evolutionary loop: reason=<unable to generate enough novel mutants>"
            )
            return population

        population = next_gen

    logger.info(
        "stopping evolutionary loop: reasoin=<reached maximum number of generations>"
    )
    return population
