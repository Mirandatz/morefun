import pathlib
import typing

import keras
import tensorflow as tf
import typer

import gge.experiments.gecco.run_evolution as run_exp

DataGen: typing.TypeAlias = keras.preprocessing.image.DirectoryIterator


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


def main(
    model_path: pathlib.Path = typer.Option(
        ...,
        "-m",
        "--model-path",
        file_okay=False,
        dir_okay=True,
        readable=True,
        exists=True,
    ),
    dataset_dir: pathlib.Path = typer.Option(
        ...,
        "-d",
        "--dataset",
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
    ),
) -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    model: keras.Model = keras.models.load_model(model_path)
    test_data = get_test_data_gen(
        dataset_dir=dataset_dir,
        batch_size=run_exp.BATCH_SIZE,
        input_shape=(run_exp.IMAGE_WIDTH, run_exp.IMAGE_HEIGHT),
    )

    model.evaluate(test_data)


if __name__ == "__main__":
    typer.run(main)
