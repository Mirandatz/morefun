import pathlib
import pickle

from loguru import logger

import gge.composite_genotypes as cg

GENOTYPE_EXTENSION = ".genotype"


def save_genotype_to_directory(
    genotype: cg.CompositeGenotype,
    directory: pathlib.Path,
) -> None:
    assert directory.is_dir()

    filename = directory / f"{genotype.unique_id.hex}{GENOTYPE_EXTENSION}"
    serialized = serialize_genotype(genotype)
    filename.write_bytes(serialized)


def save_population_genotypes(
    population: list[cg.CompositeGenotype],
    directory: pathlib.Path,
) -> None:
    assert directory.is_dir()

    logger.info("started saving population genotypes")

    saved = 0

    for genotype in population:
        save_genotype_to_directory(genotype, directory)
        saved += 1

    if saved == 0:
        logger.warning("`population` was empty, no genotypes were saved")

    logger.info(
        f"finished saving population genotypes, number of saved genotypes=<{saved}>"
    )


def serialize_genotype(genotype: cg.CompositeGenotype) -> bytes:
    return pickle.dumps(genotype, protocol=pickle.HIGHEST_PROTOCOL)
