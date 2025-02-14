#!/bin/env python

import pathlib
from pathlib import Path
from typing import Annotated

import pandas as pd
import tensorflow as tf
import typer
from keras import Model as KerasModel
from loguru import logger

import morefun.evolutionary.fitnesses as gf
import morefun.evolutionary.generations
import morefun.evolutionary.novelty
import morefun.paths
import morefun.phenotypes
import morefun.randomness
import morefun.redirection
from morefun.evolutionary.generations import EvaluatedGenotype
from morefun.experiments.settings import (
    MorefunSettings,
    configure_logger,
    configure_tensorflow,
    load_morefun_settings,
)


def get_trained_model(
    individual: EvaluatedGenotype,
    output_dir: Path,
    settings: MorefunSettings,
) -> KerasModel:
    weights_path = morefun.paths.get_model_weights_path(
        output_dir=output_dir,
        uuid=individual.genotype.unique_id,
    )

    model: KerasModel = gf.make_classification_model(
        phenotype=individual.phenotype,
        input_shape=settings.dataset.input_shape,
        class_count=settings.dataset.class_count,
    )

    if weights_path.exists():
        model.load_weights(weights_path)
        return model

    train = gf.load_train_partition(
        input_shape=settings.dataset.input_shape,
        batch_size=settings.final_train.batch_size,
        directory=settings.dataset.get_and_check_train_dir(),
    )

    test = gf.load_non_train_partition(
        input_shape=settings.dataset.input_shape,
        batch_size=settings.final_train.batch_size,
        directory=settings.dataset.get_and_check_test_dir(),
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=settings.final_train.early_stop_patience,
        restore_best_weights=True,
        monitor="loss",
    )

    model.fit(
        train,
        validation_data=test,
        epochs=999,
        callbacks=[early_stop],
    )

    model.save_weights(
        morefun.paths.get_model_weights_path(
            output_dir=output_dir,
            uuid=individual.genotype.unique_id,
        )
    )

    return model


def individual_to_dataframe_row(
    individual: morefun.evolutionary.generations.EvaluatedGenotype,
    output_dir: pathlib.Path,
    settings: MorefunSettings,
) -> dict[str, float | int | str]:
    logger.info(f"processing genotype=<{individual.genotype.unique_id.hex}>")

    trained_model = get_trained_model(individual, output_dir, settings)

    train = gf.load_non_train_partition(
        input_shape=settings.dataset.input_shape,
        batch_size=settings.final_train.batch_size,
        directory=settings.dataset.get_and_check_train_dir(),
    )

    test = gf.load_non_train_partition(
        input_shape=settings.dataset.input_shape,
        batch_size=settings.final_train.batch_size,
        directory=settings.dataset.get_and_check_test_dir(),
    )

    _, train_accuracy = trained_model.evaluate(train, verbose=0)
    _, test_accuracy = trained_model.evaluate(test, verbose=0)

    return {
        "uuid": individual.genotype.unique_id.hex,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "num_params": trained_model.count_params(),
    }


def main(
    settings_path: Annotated[
        Path,
        typer.Option(
            "-s",
            "--settings-path",
            file_okay=True,
            exists=True,
            readable=True,
            dir_okay=False,
        ),
    ],
) -> None:
    settings = load_morefun_settings(settings_path)
    configure_logger(settings.output)
    configure_tensorflow(settings.tensorflow)

    ouptut_dir = settings.output.directory
    csv_path = ouptut_dir / "analysis.csv"

    generation_numbers = morefun.paths.get_generation_numbers(ouptut_dir)

    assert generation_numbers, "No generations found"

    gen_to_use = min(max(generation_numbers), 51)

    last_checkpoint = morefun.evolutionary.generations.GenerationCheckpoint.load(
        morefun.paths.get_generation_checkpoint_path(
            output_dir=ouptut_dir,
            generation_number=gen_to_use,
        )
    )

    rows = [
        individual_to_dataframe_row(individual, ouptut_dir, settings)
        for individual in last_checkpoint.get_population()
    ]

    df = pd.DataFrame(rows)
    df.to_csv(csv_path)


if __name__ == "__main__":
    typer.run(main)
