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
import morefun.experiments.settings as gset
import morefun.paths
import morefun.phenotypes
import morefun.randomness
import morefun.redirection


def get_gitignored_dir() -> pathlib.Path:
    root = morefun.paths.get_project_root_dir()
    return root / "gge" / "playground" / "gitignored"


def get_trained_model(
    individual: morefun.evolutionary.generations.EvaluatedGenotype,
    output_dir: pathlib.Path,
    settings: gset.MorefunSettings,
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
    settings: gset.MorefunSettings,
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
    run_dirs: Annotated[
        list[Path],
        typer.Option(
            "--run-dir",
            "-r",
            dir_okay=True,
            file_okay=False,
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    for d in run_dirs:
        print(type(d), d)

    # for run_id in ["fm_0", "fm_1", "fm_2", "fm_3", "fm_4"]:
    #     run_dir = get_gitignored_dir() / "cifar10" / run_id
    #     output_dir = run_dir / "output"
    #     settings_path = run_dir / "settings.yaml"

    #     settings = gset.load_morefun_settings(settings_path)
    #     gset.configure_logger(settings.output)
    #     gset.configure_tensorflow(settings.tensorflow)

    #     last_checkpoint = morefun.evolutionary.generations.GenerationCheckpoint.load(
    #         morefun.paths.get_generation_checkpoint_path(
    #             output_dir=output_dir,
    #             generation_number=51,
    #         )
    #     )

    #     rows = [
    #         individual_to_dataframe_row(individual, output_dir, settings)
    #         for individual in last_checkpoint.get_population()
    #     ]

    #     df = pd.DataFrame(rows)
    #     df.to_csv(output_dir / f"report_{run_id}.csv")


if __name__ == "__main__":
    typer.run(main)
