import pathlib
import typing

import keras
import pandas as pd
import tensorflow as tf
import typer
from loguru import logger

import gge.experiments.gecco.run_evolution as run_exp

DataGen: typing.TypeAlias = keras.preprocessing.image.DirectoryIterator

EARLY_STOP_PATIENCE = 12


def configure_tensorflow() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def get_test_data_gen(
    dataset_dir: pathlib.Path,
    batch_size: int,
    input_shape: tuple[int, int],
) -> DataGen:
    test_data_gen = keras.preprocessing.image.ImageDataGenerator()
    return test_data_gen.flow_from_directory(
        dataset_dir / "test",
        batch_size=batch_size,
        target_size=input_shape,
    )


def get_checkpoint_path(run_dir: pathlib.Path) -> pathlib.Path:
    contents = list(run_dir.iterdir())
    assert len(contents) == 1
    return contents[0]


def get_test_accuracy(
    checkpoints_dir: pathlib.Path, dataset_dir: pathlib.Path
) -> float:
    checkpoin_path = get_checkpoint_path(checkpoints_dir)
    model: keras.Model = keras.models.load_model(checkpoin_path)
    test_data = get_test_data_gen(
        dataset_dir=dataset_dir,
        batch_size=run_exp.BATCH_SIZE,
        input_shape=(run_exp.IMAGE_WIDTH, run_exp.IMAGE_HEIGHT),
    )

    test_accuracy = model.evaluate(test_data, return_dict=True)["accuracy"]
    assert isinstance(test_accuracy, float)
    return test_accuracy


def get_dataframe_entry(
    evo_run: pathlib.Path,
    retrain_run: pathlib.Path,
    dataset_dir: pathlib.Path,
) -> dict[str, str | int | float]:
    logger.info(f"processing evo_run=<{evo_run.name}>")
    test_accuracy = get_test_accuracy(retrain_run, dataset_dir)

    return {
        "evo_run": evo_run.name,
        "retrain_run": retrain_run.name,
        "test_acc": test_accuracy,
    }


def main(
    results_dir: pathlib.Path = typer.Option(..., "-i"),
    dataset_dir: pathlib.Path = typer.Option(
        ..., "-d", file_okay=False, dir_okay=True, exists=True, readable=True
    ),
    output_filename: pathlib.Path = typer.Option(..., "-o"),
) -> None:
    configure_tensorflow()

    data = []
    for evo_run in results_dir.iterdir():
        for retrain_run in evo_run.iterdir():
            datum = get_dataframe_entry(evo_run, retrain_run, dataset_dir)
            data.append(datum)

    df = pd.DataFrame(data)
    df.to_parquet(output_filename)


if __name__ == "__main__":
    typer.run(main)
