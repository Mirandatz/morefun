#!/bin/env python


import pandas as pd
import plotly.express as px

from morefun.paths import get_project_root_dir


def main() -> None:
    experiment_dir = (
        get_project_root_dir() / "persistent" / "experiments" / "cifar10_train_val_test"
    )
    df = pd.read_csv(experiment_dir / "final_results.csv")
    df["run_nr"] = df["run"].str.split("_").str[-1].astype("category")
    px.scatter(
        df,
        x="num_params",
        y="validation_loss",
        color="run_nr",
        labels={"num_params": "Params", "validation_loss": "Loss (validation)"},
    ).show()


if __name__ == "__main__":
    main()
