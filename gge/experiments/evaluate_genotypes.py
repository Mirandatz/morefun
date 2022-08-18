import pathlib

from loguru import logger

import gge.composite_genotypes as cg
import gge.fitnesses as cf
import gge.persistence as gp


def evaluate_single_genotype(
    genotype: cg.CompositeGenotype,
    params: cf.FitnessEvaluationParameters,
    output_dir: pathlib.Path,
) -> cf.FitnessEvaluationResult:
    logger.info(f"starting fitness evaluation of genotype=<{genotype}>")

    assert output_dir.is_dir()

    fitness_evaluation_result = cf.evaluate(genotype, params)

    gp.save_fitness_evaluation_result_to_directory(
        fitness_evaluation_result,
        output_dir,
    )

    logger.info(f"finished fitness evaluation of genotype=<{genotype}>")

    return fitness_evaluation_result


def evaluate_population(
    genotypes: list[cg.CompositeGenotype],
    evaluation_params: cf.FitnessEvaluationParameters,
    output_dir: pathlib.Path,
) -> None:
    if not output_dir.is_dir():
        raise ValueError(f"`output_dir` is not a directory, path=<{output_dir}>")

    for index, genotype in enumerate(genotypes):
        logger.info(f"evaluating genotype: {index}/{len(genotypes)}")
        evaluate_single_genotype(
            genotype,
            evaluation_params,
            output_dir=output_dir,
        )
