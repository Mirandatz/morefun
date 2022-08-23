import pathlib
import pickle

import attrs
from loguru import logger

import gge.fitnesses as cf
import gge.novelty
import gge.randomness as rand

GENERATION_OUTPUT_EXTENSION = ".gen_out"


@attrs.define
class GenerationOutput:
    generation_number: int
    fittest: list[cf.FitnessEvaluationResult]
    novelty_tracker: gge.novelty.NoveltyTracker
    rng: rand.RNG


def get_generation_output_path(
    generation_number: int,
    directory: pathlib.Path,
) -> pathlib.Path:
    return directory / f"{generation_number}.gen_out"


def save_generation_output(
    generation_number: int,
    fittest: list[cf.FitnessEvaluationResult],
    rng: rand.RNG,
    novelty_tracker: gge.novelty.NoveltyTracker,
    output_dir: pathlib.Path,
) -> None:
    logger.info(
        f"started saving generation output, generation_number=<{generation_number}>"
    )

    if len(fittest) == 0:
        raise ValueError("`fittest` is empty")

    gen_out = GenerationOutput(
        generation_number=generation_number,
        fittest=fittest,
        novelty_tracker=novelty_tracker,
        rng=rng,
    )

    blob = pickle.dumps(gen_out, protocol=pickle.HIGHEST_PROTOCOL)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = get_generation_output_path(generation_number, output_dir)
    path.write_bytes(blob)

    logger.info(
        f"finished saving generation output, generation_number=<{generation_number}>"
    )


def load_generation_output(
    generation_number: int,
    directory: pathlib.Path,
) -> GenerationOutput:
    path = get_generation_output_path(generation_number, directory)
    deserialized = pickle.loads(path.read_bytes())
    assert isinstance(deserialized, GenerationOutput)
    return deserialized
