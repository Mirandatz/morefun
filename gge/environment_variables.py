"""
Our implementation assumes that it will be executed in a purpose-built container,
but to achieve some flexibility, we use environment variables instead of constants to
identify where data should be read from and where data should be written.

This module contains functions that read and parse such environment variables.
"""

import os
import pathlib

import attrs

# TODO: improve validation and error messages


@attrs.frozen
class GgePaths:
    grammar_path: pathlib.Path
    initial_population_dir: pathlib.Path

    train_dataset_dir: pathlib.Path
    validation_dataset_dir: pathlib.Path
    test_dataset_dir: pathlib.Path

    output_dir: pathlib.Path
    logging_dir: pathlib.Path
    code_dir: pathlib.Path


def get_paths() -> GgePaths:
    return GgePaths(
        grammar_path=pathlib.Path(
            os.getenv(
                key="GGE_GRAMMAR",
                default="/gge/grammar.lark",
            )
        ),
        initial_population_dir=pathlib.Path(
            os.getenv(
                key="GGE_INITIAL_POPULATION",
                default="/gge/initial_population",
            )
        ),
        train_dataset_dir=pathlib.Path(
            os.getenv(
                key="GGE_TRAIN_DIR",
                default="/gge/dataset/train",
            )
        ),
        validation_dataset_dir=pathlib.Path(
            os.getenv(
                key="GGE_VALIDATION_DIR",
                default="/gge/dataset/validation",
            )
        ),
        test_dataset_dir=pathlib.Path(
            os.getenv(
                key="GGE_TEST_DIR",
                default="/gge/dataset/test",
            )
        ),
        output_dir=pathlib.Path(
            os.getenv(
                key="GGE_OUTPUT_DIR",
                default="/gge/output",
            )
        ),
        logging_dir=pathlib.Path(
            os.getenv(
                key="GGE_LOGGING_DIR",
                default="/gge/logging",
            )
        ),
        code_dir=pathlib.Path(
            os.getenv(
                key="GGE_CODE_DIR",
                default="/gge/gge",
            )
        ),
    )


def get_rng_seed() -> int:
    envvar = "GGE_RNG_SEED"
    raw = os.getenv(envvar, None)
    if raw is None:
        raise ValueError(f"environment variable not defined, name=<{envvar}>")

    return int(raw)
