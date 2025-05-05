#!/bin/env python

import pathlib
from pathlib import Path

import pandas as pd
import tensorflow as tf
from keras import Model as KerasModel
from loguru import logger

import morefun.evolutionary.fitnesses as gf
import morefun.evolutionary.generations
import morefun.paths
from morefun.evolutionary.generations import EvaluatedGenotype
from morefun.experiments.settings import (
    MorefunSettings,
    configure_logger,
    configure_tensorflow,
    load_morefun_settings,
)
from morefun.paths import get_project_root_dir

CsvRow = dict[str, float | int | str]


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

    try:
        validation = gf.load_non_train_partition(
            input_shape=settings.dataset.input_shape,
            batch_size=settings.final_train.batch_size,
            directory=settings.dataset.get_and_check_validation_dir(),
        )

        validation_loss, validation_accuracy = trained_model.evaluate(
            validation, verbose=0
        )

    except Exception as e:
        logger.error(f"failed to evaluate validation set: {e}")
        validation_loss, validation_accuracy = float("nan"), float("nan")

    test = gf.load_non_train_partition(
        input_shape=settings.dataset.input_shape,
        batch_size=settings.final_train.batch_size,
        directory=settings.dataset.get_and_check_test_dir(),
    )

    train_loss, train_accuracy = trained_model.evaluate(train, verbose=0)
    test_loss, test_accuracy = trained_model.evaluate(test, verbose=0)

    return {
        "uuid": individual.genotype.unique_id.hex,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "num_params": trained_model.count_params(),
    }


def make_csv_rows_for_run(run_dir: Path) -> CsvRow:
    settings_path = run_dir / "settings.yaml"

    print(f"processing run directory: {run_dir}")

    settings = load_morefun_settings(settings_path)
    configure_logger(settings.output)
    configure_tensorflow(settings.tensorflow)

    ouptut_dir = settings.output.directory

    generation_numbers = morefun.paths.get_generation_numbers(ouptut_dir)

    if not generation_numbers:
        raise ValueError(
            f"no generation numbers found in output directory: {ouptut_dir}"
        )

    gen_to_use = min(max(generation_numbers), 51)

    last_checkpoint = morefun.evolutionary.generations.GenerationCheckpoint.load(
        morefun.paths.get_generation_checkpoint_path(
            output_dir=ouptut_dir,
            generation_number=gen_to_use,
        )
    )

    return [
        individual_to_dataframe_row(individual, ouptut_dir, settings)
        for individual in last_checkpoint.get_population()
    ]


def main() -> None:
    experiment_dir = (
        get_project_root_dir() / "persistent" / "experiments" / "cifar10_train_val_test"
    )

    run_dirs = sorted([d for d in experiment_dir.iterdir() if d.is_dir()])
    if not run_dirs:
        raise ValueError(
            f"no run directories found in experiment directory:{experiment_dir}"
        )

    dfs = []
    for run_dir in run_dirs:
        df = pd.DataFrame(make_csv_rows_for_run(run_dir))
        df["run"] = run_dir.stem
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(
        experiment_dir / "final_results.csv",
        index=False,
    )
    print("done")


if __name__ == "__main__":
    main()
