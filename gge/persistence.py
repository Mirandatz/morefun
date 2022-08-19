import pathlib
import pickle

from loguru import logger

import gge.composite_genotypes as cg
import gge.fitnesses as cf

GENOTYPE_EXTENSION = ".genotype"
FITNESS_EVALUATION_RESULT_EXTENSION = ".fitness_evaluation_result"


def save_genotype(genotype: cg.CompositeGenotype, path: pathlib.Path) -> None:
    """
    Creates `path` parents, then serializes `genotype` and writes it to `path`.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = serialize_genotype(genotype)
    path.write_bytes(serialized)

    logger.info(f"saved genotype=<{genotype}>, path=<{path}>")


def save_fitness_evaluation_result(
    fer: cf.FitnessEvaluationResult,
    path: pathlib.Path,
) -> None:
    """
    Creates `path` parents, then serializes `fer` and writes it to `path`.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = serialize_fitness_evaluation_result(fer)
    path.write_bytes(serialized)

    logger.info(
        f"saved fitness evaluation result of genotype=<{fer.genotype}>, path=<{path}>"
    )


def get_generation_dir(generation: int, base_output_dir: pathlib.Path) -> pathlib.Path:
    assert generation >= 0

    generation_dir = base_output_dir / str(generation)
    generation_dir.mkdir(parents=True, exist_ok=True)
    return generation_dir


def get_genotypes_dir(generation: int, base_output_dir: pathlib.Path) -> pathlib.Path:
    assert generation >= 0

    generation_dir = get_generation_dir(generation, base_output_dir)
    genotypes_dir = generation_dir / "genotypes"
    genotypes_dir.mkdir(parents=True, exist_ok=True)
    return genotypes_dir


def get_fitness_evaluation_results_dir(
    generation: int, base_output_dir: pathlib.Path
) -> pathlib.Path:
    assert generation >= 0

    generation_dir = get_generation_dir(generation, base_output_dir)
    fers_dir = generation_dir / "fitness_evaluation_results"
    fers_dir.mkdir(parents=True, exist_ok=True)
    return fers_dir


def get_genotype_path(
    genotype: cg.CompositeGenotype,
    generation: int,
    base_output_dir: pathlib.Path,
) -> pathlib.Path:
    assert generation >= 0

    genotypes_dir = get_genotypes_dir(generation, base_output_dir)
    return genotypes_dir / f"{genotype.unique_id.hex}{GENOTYPE_EXTENSION}"


def save_generation_genotypes(
    genotypes: list[cg.CompositeGenotype],
    generation: int,
    base_output_dir: pathlib.Path,
) -> None:
    assert generation >= 0

    logger.info(
        f"started saving genotypes for generation=<{generation}>, count=<{len(genotypes)}>"
    )

    for genotype in genotypes:
        path = get_genotype_path(genotype, generation, base_output_dir)
        save_genotype(genotype, path)

    logger.info("finished saving population genotypes")


def save_generation_fitness_evaluation_results(
    fers: list[cf.FitnessEvaluationResult],
    generation: int,
    base_output_dir: pathlib.Path,
) -> None:
    assert generation >= 0

    logger.info(
        f"started saving fitness evaluation results for generation=<{generation}>, count=<{len(fers)}>"
    )

    for fer in fers:
        path = get_fitness_evaluation_result_path(fer, generation, base_output_dir)
        save_fitness_evaluation_result(fer, path)

    logger.info("finished saving fitness evaluation results")


def get_fitness_evaluation_result_path(
    fer: cf.FitnessEvaluationResult,
    generation: int,
    base_output_dir: pathlib.Path,
) -> pathlib.Path:
    assert generation >= 0

    fers_dir = get_fitness_evaluation_results_dir(generation, base_output_dir)
    return (
        fers_dir / f"{fer.genotype.unique_id.hex}{FITNESS_EVALUATION_RESULT_EXTENSION}"
    )


def load_genotype(path: pathlib.Path) -> cg.CompositeGenotype:
    return deserialize_genotype(path.read_bytes())


def serialize_genotype(genotype: cg.CompositeGenotype) -> bytes:
    return pickle.dumps(genotype, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_genotype(serialized: bytes) -> cg.CompositeGenotype:
    genotype = pickle.loads(serialized)
    assert isinstance(genotype, cg.CompositeGenotype)
    return genotype


def serialize_fitness_evaluation_result(result: cf.FitnessEvaluationResult) -> bytes:
    return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_fitness_evaluation_result(
    serialized: bytes,
) -> cf.FitnessEvaluationResult:
    result = pickle.loads(serialized)
    assert isinstance(result, cf.SuccessfulEvaluationResult | cf.FailedEvaluationResult)
    return result


def load_fitness_evaluation_result(path: pathlib.Path) -> cf.FitnessEvaluationResult:
    return deserialize_fitness_evaluation_result(path.read_bytes())
