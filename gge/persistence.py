import pathlib
import pickle

from loguru import logger

import gge.composite_genotypes as cg
import gge.fitnesses as cf

GENOTYPE_EXTENSION = ".genotype"
FITNESS_EVALUATION_RESULT_EXTENSION = ".fitness_evaluation_result"


def save_genotype_to_directory(
    genotype: cg.CompositeGenotype,
    directory: pathlib.Path,
) -> None:
    assert directory.is_dir()

    serialized = serialize_genotype(genotype)

    path_without_extension = directory / genotype.unique_id.hex
    full_path = path_without_extension.with_suffix(GENOTYPE_EXTENSION)

    full_path.write_bytes(serialized)


def save_population_genotypes(
    population: list[cg.CompositeGenotype],
    directory: pathlib.Path,
) -> None:
    assert len(population) > 0
    assert directory.is_dir()

    logger.info("started saving population genotypes")

    for genotype in population:
        save_genotype_to_directory(genotype, directory)

    logger.info("finished saving population genotypes")


def load_genotype(path: pathlib.Path) -> cg.CompositeGenotype:
    return deserialize_genotype(path.read_bytes())


def load_population_genotypes(directory: pathlib.Path) -> list[cg.CompositeGenotype]:
    assert directory.is_dir()

    paths = list(directory.glob(f"*{GENOTYPE_EXTENSION}"))
    if len(paths) == 0:
        raise ValueError(
            f"directory=<{directory}> does not contain files with genotype extension=<{GENOTYPE_EXTENSION}>"
        )

    return [load_genotype(p) for p in paths]


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


def save_fitness_evaluation_result_to_directory(
    result: cf.FitnessEvaluationResult,
    directory: pathlib.Path,
) -> None:
    assert directory.is_dir()

    serialized = serialize_fitness_evaluation_result(result)

    path_without_extension = directory / result.genotype.unique_id.hex
    full_path = path_without_extension.with_suffix(GENOTYPE_EXTENSION)

    full_path.write_bytes(serialized)


def load_fitness_evaluation_result(path: pathlib.Path) -> cf.FitnessEvaluationResult:
    return deserialize_fitness_evaluation_result(path.read_bytes())
