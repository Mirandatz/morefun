import pathlib
import pickle
import shutil
import tempfile

import attrs
import tensorflow as tf
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


def load_generation_output(path: pathlib.Path) -> GenerationOutput:
    deserialized = pickle.loads(path.read_bytes())
    assert isinstance(deserialized, GenerationOutput)
    return deserialized


def load_latest_generation_output(output_dir: pathlib.Path) -> GenerationOutput:
    paths = output_dir.glob(f"*{GENERATION_OUTPUT_EXTENSION}")
    latest = max(paths, key=lambda path: int(path.stem))
    return load_generation_output(latest)


def save_and_zip_tf_model(model: tf.keras.Model, path: pathlib.Path) -> None:
    """
    Saves a Tensorflow model to a single file.

    Obs: The difference between this function and the one provided by Tensorflow
    is that Tensorflow's `save` function creates a directory with multiple files.
    """

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
