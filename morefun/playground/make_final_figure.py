#!/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from morefun.paths import get_project_root_dir


def main() -> None:
    experiment_dir = (
        get_project_root_dir()
        / "persistent/scenario_with_validation/experiments/cifar10_train_val_test"
    )
    csv_path = experiment_dir / "final_results.csv"
    val_png_path = experiment_dir / "final_result_val_loss.png"
    test_png_path = experiment_dir / "final_result_test_loss.png"

    df = pd.read_csv(csv_path)
    df["run_nr"] = df["run"].str.split("_").str[-1].astype("category")

    print(df.columns)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df,
        x="num_params",
        y="validation_loss",
        hue="run_nr",
        legend="full",
        ax=ax,
    )
    ax.set_xlabel("Params")
    ax.set_ylabel("Loss (validation)")
    fig.tight_layout()
    fig.savefig(val_png_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df,
        x="num_params",
        y="train_loss",
        hue="run_nr",
        legend="full",
        ax=ax,
    )
    ax.set_xlabel("Params")
    ax.set_ylabel("Loss (train)")
    fig.tight_layout()
    fig.savefig(test_png_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
