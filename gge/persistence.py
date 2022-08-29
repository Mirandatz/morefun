import pathlib
import pickle
import shutil
import tempfile

import attrs
import pandas as pd
import tensorflow as tf
from loguru import logger

import gge.composite_genotypes as cg
import gge.fitnesses as gf
import gge.novelty
import gge.randomness as rand

GENERATION_OUTPUT_EXTENSION = ".gen_out"
TRAINING_HISTORY_EXTENSION = ".train_hist"
GENOTYPE_EXTENSION = ".genotype"
MODEL_EXTENSION = ".zipped_tf_model"


@attrs.define
class GenerationOutput:
    generation_number: int
    fittest: list[gf.FitnessEvaluationResult]
    novelty_tracker: gge.novelty.NoveltyTracker
    rng: rand.RNG


def get_generational_artifacts_path(
    generation_number: int,
    directory: pathlib.Path,
) -> pathlib.Path:
    return (directory / str(generation_number)).with_suffix(GENERATION_OUTPUT_EXTENSION)


def get_genotype_path(
    genotype: cg.CompositeGenotype,
    dir: pathlib.Path,
) -> pathlib.Path:
    return (dir / genotype.unique_id.hex).with_suffix(GENOTYPE_EXTENSION)


def get_model_path(genotype_uuid_hex: str, dir: pathlib.Path) -> pathlib.Path:
    return (dir / genotype_uuid_hex).with_suffix(MODEL_EXTENSION)


def get_training_history_path(
    genotype_uuid_hex: str, dir: pathlib.Path
) -> pathlib.Path:
    return (dir / genotype_uuid_hex).with_suffix(TRAINING_HISTORY_EXTENSION)


def save_generational_artifacts(
    generation_number: int,
    fittest: list[gf.FitnessEvaluationResult],
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
    path = get_generational_artifacts_path(generation_number, output_dir)
    path.write_bytes(blob)

    logger.info(
        f"finished saving generation output, generation_number=<{generation_number}>"
    )


def load_generational_artifacts(path: pathlib.Path) -> GenerationOutput:
    deserialized = pickle.loads(path.read_bytes())
    assert isinstance(deserialized, GenerationOutput)
    return deserialized


def load_latest_generational_artifacts(output_dir: pathlib.Path) -> GenerationOutput:
    paths = output_dir.glob(f"*{GENERATION_OUTPUT_EXTENSION}")
    latest = max(paths, key=lambda path: int(path.stem))
    return load_generational_artifacts(latest)


def save_and_zip_tf_model(model: tf.keras.Model, path: pathlib.Path) -> None:
    """
    Saves a Tensorflow model to a single file.

    Obs: The difference between this function and the one provided by Tensorflow
    is that Tensorflow's `save` function creates a directory with multiple files.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir="/dev/shm") as _dir:
        tmp_dir = pathlib.Path(_dir)

        model_dir = tmp_dir / "tf_model"

        tf.keras.models.save_model(
            model=model,
            filepath=model_dir,
            include_optimizer=True,
            save_format="tf",
            save_traces=True,
        )

        zipped = shutil.make_archive(
            base_name=str(tmp_dir / "tf_model_zipped"),
            format="zip",
            root_dir=model_dir,
            base_dir=".",
        )

        shutil.move(src=zipped, dst=path)


def load_zipped_tf_model(path: pathlib.Path) -> tf.keras.Model:
    with tempfile.TemporaryDirectory(dir="/dev/shm") as _dir:
        tmp_dir = pathlib.Path(_dir)

        shutil.unpack_archive(filename=path, extract_dir=tmp_dir, format="zip")

        return tf.keras.models.load_model(tmp_dir)


def save_genotype(genotype: cg.CompositeGenotype, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(genotype, protocol=pickle.HIGHEST_PROTOCOL))


def save_training_history(hist: gf.TrainingHistory, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(hist, protocol=pickle.HIGHEST_PROTOCOL))
