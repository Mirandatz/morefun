"""
Our implementation assumes that it will be executed in a purpose-built container,
but to achieve some flexibility, we use environment variables (instead of constants)
to identify where data should be read from and where data should be written.

This module contains `Option` objects (from the `typer` library) that can be used
to parse command line arguments and environment variables.
"""

import pathlib
import sys

import typer
from loguru import logger

import gge.redirection

# TODO: improve validation and error messages


RNG_SEED = typer.Option(..., "--rng-seed", envvar="GGE_RNG_SEED")

MAX_GENERATIONS = typer.Option(..., envvar="GGE_MAX_GENERATIONS", min=1)
MUTANTS_PER_GENERATION = typer.Option(..., envvar="GGE_MUTANTS_PER_GENERATION", min=1)
MAX_FAILURES = typer.Option(500, envvar="GGE_MAX_FAILURES", min=1)
BATCH_SIZE = typer.Option(..., envvar="GGE_BATCH_SIZE", min=1)
EPOCHS = typer.Option(..., envvar="GGE_EPOCHS", min=1)

IMAGE_WIDTH = typer.Option(..., envvar="GGE_IMG_WIDTH", min=1)
IMAGE_HEIGHT = typer.Option(..., envvar="GGE_IMG_HEIGHT", min=1)
IMAGE_DEPTH = typer.Option(..., envvar="GGE_IMG_DEPTH", min=1)
CLASS_COUNT = typer.Option(..., envvar="GGE_CLASS_COUNT", min=1)

LOG_LEVEL = typer.Option("INFO", envvar="GGE_LOG_LEVEL")

EXPECTED_NUMBER_OF_TRAIN_INSTANCES = typer.Option(
    None, envvar="GGE_EXPECTED_NUMBER_OF_TRAIN_INSTANCES"
)
EXPECTED_NUMBER_OF_VALIDATION_INSTANCES = typer.Option(
    None, envvar="GGE_EXPECTED_NUMBER_OF_VALIDATION_INSTANCES"
)
EXPECTED_NUMBER_OF_TEST_INSTANCES = typer.Option(
    None, envvar="GGE_EXPECTED_NUMBER_OF_TEST_INSTANCES"
)

GGE_DIR = pathlib.Path("/gge")

GRAMMAR_PATH = typer.Option(
    GGE_DIR / "grammar.lark",
    envvar="GGE_GRAMMAR_PATH",
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
)

INITIAL_POPULATION_DIR = typer.Option(
    GGE_DIR / "initial_population",
    envvar="GGE_INITIAL_POPULATION_DIR",
    dir_okay=True,
    exists=True,
    readable=True,
    file_okay=False,
)

TRAIN_DATASET_DIR = typer.Option(
    GGE_DIR / "dataset" / "train",
    envvar="GGE_TRAIN_DIR",
    dir_okay=True,
    exists=True,
    readable=True,
    file_okay=False,
)

VALIDATION_DATASET_DIR = typer.Option(
    GGE_DIR / "dataset" / "validation",
    envvar="GGE_VALIDATION_DIR",
    dir_okay=True,
    exists=True,
    readable=True,
    file_okay=False,
)

TEST_DATASET_DIR = typer.Option(
    GGE_DIR / "dataset" / "test",
    envvar="GGE_TEST_DIR",
    dir_okay=True,
    exists=True,
    readable=True,
    file_okay=False,
)

OUTPUT_DIR = typer.Option(
    GGE_DIR / "output",
    envvar="GGE_OUTPUT_DIR",
    dir_okay=True,
    exists=True,
    readable=True,
    writable=True,
    file_okay=False,
)

LOG_DIR = typer.Option(
    GGE_DIR / "log",
    envvar="GGE_LOG_DIR",
    dir_okay=True,
    exists=True,
    readable=True,
    writable=True,
    file_okay=False,
)


def configure_logger(log_dir: pathlib.Path, log_level: str) -> None:
    if log_level not in ["DEBUG", "INFO", "WARNING"]:
        raise ValueError(f"unknown log_level=<{log_level}>")

    if not log_dir.is_dir():
        raise ValueError(f"log_dir is not a directory, path=<{log_dir}>")

    logger.remove()
    logger.add(sink=sys.stderr, level=log_level)
    logger.add(sink=log_dir / "log.txt")

    # checking if we can write to both sinks
    logger.info("started logger")


def validate_dataset_dir(
    path: pathlib.Path,
    img_height: int,
    img_width: int,
    expected_num_instances: int,
    expected_class_count: int,
) -> None:
    import tensorflow as tf

    with gge.redirection.discard_stderr_and_stdout():
        with tf.device("cpu"):
            ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
                directory=path,
                batch_size=None,
                image_size=(img_height, img_width),
                label_mode="categorical",
                shuffle=False,
                color_mode="rgb",
            )

            num_instances = ds.cardinality().numpy()
            if num_instances != expected_num_instances:
                raise ValueError(
                    f"unexpected number of instances. found=<{num_instances}>, expected=<{expected_num_instances}>"
                )

            num_classes = len(ds.class_names)
            if num_classes != expected_class_count:
                raise ValueError(
                    f"unexpected number of classes. found=<{num_classes}>, expected=<{expected_class_count}>"
                )
