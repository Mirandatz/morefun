import datetime as dt
import pathlib
import pickle

import pandas as pd

import gge.fitnesses as fit
import gge.startup_settings as gge_settings


def parse_evaluation_result_file(
    path: pathlib.Path,
) -> dict[str, str | float | bool | dt.datetime]:
    assert path.is_file()

    evaluation_result = pickle.loads(path.read_bytes())

    match evaluation_result:
        case fit.SuccessfulEvaluationResult() as succ:
            return {
                "uuid": succ.genotype.unique_id.hex,
                "fitness": succ.fitness,
                "succeeded": True,
                "failure_description": "",
                "start_time": succ.start_time,
                "end_time": succ.end_time,
            }

        case fit.FailedEvaluationResult() as fail:
            return {
                "uuid": fail.genotype.unique_id.hex,
                "fitness": float("nan"),
                "succeeded": False,
                "failure_description": fail.description,
                "start_time": fail.start_time,
                "end_time": fail.end_time,
            }

        case _:
            raise ValueError(f"unknown evaluation result type=<{evaluation_result}>")


def process_generation_results_dir(path: pathlib.Path) -> pd.DataFrame:
    assert path.is_dir()

    fittest_dir = path / "fittest"
    evaluation_results = list(fittest_dir.iterdir())
    assert len(evaluation_results) > 0
    assert all(p.is_file() for p in evaluation_results)

    rows = [parse_evaluation_result_file(p) for p in evaluation_results]
    return pd.DataFrame.from_records(rows)


def process_experiment_results_dir(path: pathlib.Path) -> pd.DataFrame:
    assert path.is_dir()

    gen_dirs = list(path.glob("generation*"))
    assert len(gen_dirs) > 0
    assert all(gd.is_dir() for gd in gen_dirs)

    dfs = []
    for gd in gen_dirs:
        df = process_generation_results_dir(gd)
        _, gen_nr = gd.name.split("_")
        df["gen_nr"] = int(gen_nr)
        dfs.append(df)

    return pd.concat(dfs, axis="index")


def main(experiment_output_dir: pathlib.Path = gge_settings.OUTPUT_DIR) -> None:
    df = process_experiment_results_dir(experiment_output_dir)
    df.to_parquet(experiment_output_dir / "fittest.parquet")


if __name__ == "__main__":
    main()
