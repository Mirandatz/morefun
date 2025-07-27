#!/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pymoo.indicators.hv import HV

from morefun.paths import get_project_root_dir


def main() -> None:
    experiment_dir = (
        get_project_root_dir()
        / "persistent/scenario_with_validation/experiments/cifar10_train_val_test"
    )
    csv_path = experiment_dir / "cifar10_stuff_for_hypervolume.csv"

    df = pd.read_csv(csv_path)

    max_loss = df["train_loss"].max()
    min_loss = df["train_loss"].min()
    delta_loss = max_loss - min_loss

    max_num_params = df["number_of_parameters"].max()
    min_num_params = df["number_of_parameters"].min()
    delta_num_params = max_num_params - min_num_params

    rows = []
    for run_nr in sorted(df["run_nr"].unique()):
        run_df = df[df["run_nr"] == run_nr]

        for gen_nr in sorted(run_df["generation"].unique()):
            gen_df = run_df[run_df["generation"] == gen_nr]

            loss = gen_df["train_loss"].to_numpy()
            norm_loss = (loss - min_loss) / delta_loss

            num_params = gen_df["number_of_parameters"].to_numpy()
            norm_num_params = (num_params - min_num_params) / delta_num_params

            objectives = np.vstack((norm_loss, norm_num_params)).T

            hv_value = HV(ref_point=(1.1, 1.1))(objectives)
            rows.append(
                {
                    "run_nr": run_nr,
                    "gen_nr": gen_nr,
                    "hypervolume": hv_value,
                }
            )
    hv_df = pd.DataFrame(rows)
    sns.lineplot(
        data=hv_df,
        x="gen_nr",
        y="hypervolume",
        hue="run_nr",
        palette="viridis",
    )
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    # plt.legend(title="Generation", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        experiment_dir / "hypervolume_over_generations.png",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
