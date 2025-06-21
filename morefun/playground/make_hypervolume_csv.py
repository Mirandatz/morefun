#!/bin/env python

from pathlib import Path

import pandas as pd

from morefun.evolutionary.generations import EvaluatedGenotype, GenerationCheckpoint
from morefun.paths import get_project_root_dir

CsvRow = dict[str, float | int | str]


def evaluated_individual_to_csv_row(
    evaluated_individual: EvaluatedGenotype,
) -> dict[str, float | str]:
    fit = evaluated_individual.fitness.to_effective_fitnesses_dict()
    assert "TrainLoss" in fit
    assert "NumberOfParameters" in fit
    return {
        "train_loss": fit["TrainLoss"],
        "number_of_parameters": fit["NumberOfParameters"],
        "id": evaluated_individual.genotype.unique_id.hex,
    }


def generation_checkpoint_to_csv_rows(
    generation_checkpoint: GenerationCheckpoint,
) -> list[CsvRow]:
    rows = []

    evaluated_individuals = generation_checkpoint.get_population()
    assert evaluated_individuals

    for evaluated_individual in evaluated_individuals:
        row = evaluated_individual_to_csv_row(evaluated_individual)
        row["generation"] = generation_checkpoint.get_generation_number()
        rows.append(row)

    return rows


def run_directory_to_csv_rows(
    run_directory: Path,
) -> list[CsvRow]:
    results_dir = run_directory / "output"

    generation_checkpoint_paths = sorted(
        results_dir.glob("*.generation_checkpoint"), key=lambda d: d.stem
    )

    assert generation_checkpoint_paths

    rows = []
    for checkpoint_path in generation_checkpoint_paths:
        checkpoint = GenerationCheckpoint.load(checkpoint_path)
        for row in generation_checkpoint_to_csv_rows(checkpoint):
            row["run_nr"] = int(run_directory.stem.split("_")[1])
            rows.append(row)

    return rows


def main() -> None:
    experiment_dir = (
        get_project_root_dir()
        / "persistent/scenario_with_validation/experiments/cifar10_train_val_test"
    )
    csv_path = experiment_dir / "cifar10_stuff_for_hypervolume.csv"

    run_dirs = sorted(d for d in experiment_dir.glob("run_*"))
    assert run_dirs

    all_rows = []
    for run_dir in run_dirs:
        all_rows.extend(run_directory_to_csv_rows(run_dir))

    df = pd.DataFrame(all_rows)
    df = df.sort_values(by=["run_nr", "generation"])
    df = df.reset_index(drop=True)

    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
