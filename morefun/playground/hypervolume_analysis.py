import pathlib
import re

import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume

import morefun.fitnesses as gf
import morefun.persistence as gp

MAX_MODEL_SIZE = -float(500 * 1000 * 1000)


def compute_generation_hypervolume(df: pd.DataFrame) -> float:
    data = df[["validation_accuracy", "model_size"]].to_numpy()
    data[:, 0] *= -1
    ideal = np.asarray([1.0, 1.0]) * -1
    nadir = np.asarray([0.0, MAX_MODEL_SIZE]) * -1
    ref_point = nadir
    hvc = Hypervolume(
        ref_point,
        zero_to_one=True,
        ideal=ideal,
        nadir=nadir,
        norm_ref_point=True,
    )
    hypervolume = hvc.do(data)
    return float(hypervolume)


def generation_artifact_path_to_df(path: pathlib.Path) -> pd.DataFrame:
    generation_artifacts = gp.load_generational_artifacts(path)

    rows = []
    for fer in generation_artifacts.fittest:
        match fer:
            case gf.SuccessfulEvaluationResult():
                r = {
                    "genotype": fer.genotype.unique_id.hex,
                    "validation_accuracy": fer.fitness.values[0],
                    "model_size": fer.fitness.values[1],
                }

            case gf.FailedEvaluationResult():
                r = {
                    "genotype": fer.genotype.unique_id.hex,
                    "validation_accuracy": 0,
                    "model_size": MAX_MODEL_SIZE,
                }

            case _:
                raise ValueError()

        rows.append(r)

    df = pd.DataFrame(rows)
    df["generation_number"] = generation_artifacts.get_generation_number
    return df


def run_dir_to_dataframe(run_dir: pathlib.Path) -> pd.DataFrame:
    (run_number,) = re.findall(pattern=r"seed_(\d)", string=str(run_dir))

    generational_artifacts_paths = list(run_dir.rglob("*.gen_out"))
    assert len(generational_artifacts_paths) >= 1

    dfs = (generation_artifact_path_to_df(gap) for gap in generational_artifacts_paths)

    df = pd.concat(dfs, axis="index")
    df["run_number"] = int(run_number)
    return df


def main() -> None:
    cifar10_dir = pathlib.Path(
        "/workspaces/gge/gge/playground/gitignored/cifar10_results"
    )

    dfs = (run_dir_to_dataframe(p) for p in cifar10_dir.glob("*seed*"))
    df = pd.concat(dfs, axis="index")
    df.to_parquet("/workspaces/gge/gge/playground/gitignored/hv.parquet")


if __name__ == "__main__":
    main()
