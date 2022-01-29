import enum
import pathlib
import pickle
import typing

import keras
import tensorflow as tf
import typer
from loguru import logger

import gge.composite_genotypes as cg
import gge.experiments.gecco.run_evolution as run_exp
import gge.fitnesses as gfit
import gge.neural_network as gnn

DataGen: typing.TypeAlias = keras.preprocessing.image.DirectoryIterator

EARLY_STOP_PATIENCE = 12


class EarlyStopMetric(str, enum.Enum):
    val_loss = "val_loss"
    val_accuracy = "val_accuracy"


def get_train_and_val(
    dataset_dir: pathlib.Path,
    batch_size: int,
    input_shape: tuple[int, int],
    shuffle_seed: int,
) -> tuple[DataGen, DataGen]:
    train_data_gen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    train = train_data_gen.flow_from_directory(
        dataset_dir / "train",
        batch_size=batch_size,
        target_size=input_shape,
        shuffle=True,
        seed=shuffle_seed,
    )

    val_data_gen = keras.preprocessing.image.ImageDataGenerator()
    val = val_data_gen.flow_from_directory(
        dataset_dir / "val",
        batch_size=batch_size,
        target_size=input_shape,
    )

    return train, val


def main(
    genotype_path: pathlib.Path = typer.Option(
        ...,
        "-g",
        "--genotype",
        file_okay=True,
        dir_okay=False,
        readable=True,
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
    output_dir: pathlib.Path = typer.Option(
        ...,
        "-o",
        "--output",
    ),
) -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        logger.error(f"output dir is not empty=<{output_dir}>")
        exit(-1)

    genotype: cg.CompositeGenotype = pickle.loads(genotype_path.read_bytes())
    network = gnn.make_network(
        genotype, run_exp.get_grammar(), run_exp.get_input_layer()
    )

    model = gfit.make_tf_model(network, run_exp.CLASS_COUNT)

    checkpoint = keras.callbacks.ModelCheckpoint(
        save_best_only=True,
        filepath=output_dir / "checkpoint",
        monitor="val_loss",
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
    )

    train, val = get_train_and_val(
        dataset_dir=dataset_dir,
        batch_size=run_exp.BATCH_SIZE,
        input_shape=(run_exp.IMAGE_WIDTH, run_exp.IMAGE_HEIGHT),
        shuffle_seed=0,
    )

    model.fit(
        train,
        epochs=123456,
        steps_per_epoch=train.samples // run_exp.BATCH_SIZE,
        validation_data=val,
        validation_steps=val.samples // run_exp.BATCH_SIZE,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )


if __name__ == "__main__":
    typer.run(main)
