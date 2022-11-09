import pathlib

import pandas as pd

import gge.fitnesses as gf
import gge.persistence as gp


def generation_artifacts_to_dataframe(
    generation_artifact: gp.GenerationOutput,
) -> pd.DataFrame:

    rows = []
    for fer in generation_artifact.fittest:
        match fer:
            case gf.SuccessfulEvaluationResult():
                r = {
                    "genotype": fer.genotype.unique_id.hex,
                    "validation_accuracy": fer.fitness.values[0],
                    "model_size": -1 * fer.fitness.values[1],
                }

            case gf.FailedEvaluationResult():
                r = {
                    "genotype": fer.genotype.unique_id.hex,
                    "validation_accuracy": float("-inf"),
                    "model_size": float("+inf"),
                }

            case _:
                raise ValueError()

        rows.append(r)

    return pd.DataFrame(rows)


def main() -> None:
    gen_art = gp.load_generational_artifacts(
        pathlib.Path(
            "/workspaces/gge/gge/playground/gitignored/cifar10_results/seed_0/output/0.gen_out"
        )
    )
    df = generation_artifacts_to_dataframe(gen_art)
    print(df.head())


if __name__ == "__main__":
    main()
