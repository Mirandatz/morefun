import itertools
from pathlib import Path
from typing import Optional

import pandas as pd

from morefun.evolutionary.generations import EvaluatedGenotype, GenerationCheckpoint


def load_genotypes(generation_checkpoint_path: Path) -> tuple[EvaluatedGenotype, ...]:
    return GenerationCheckpoint.load(generation_checkpoint_path).get_population()


def extract_column_names(
    old_column_names: Optional[list[str]],
    genotype: EvaluatedGenotype,
) -> list[str]:
    new_column_names = [
        "run_nr",
        " gen_nr",
        "unique_id",
        *genotype.fitness.metric_names(),
    ]
    assert (old_column_names is None) or (new_column_names == old_column_names)
    return new_column_names


def make_df_row(
    run_nr: int,
    gen_nr: int,
    genotype: EvaluatedGenotype,
) -> list[int | str | float]:
    return [
        run_nr,
        gen_nr,
        genotype.genotype.unique_id.hex,
        *genotype.fitness.effective_values(),
    ]


def main() -> None:
    persistent_dir = Path("/workspaces/morefun/persistent")

    num_runs = 5
    num_generations = 51

    df_rows = []
    column_names: Optional[list[str]] = None

    for run_nr, gen_nr in itertools.product(range(num_runs), range(num_generations)):
        generation_checkpoint_path = (
            persistent_dir
            / "v2"
            / f"run_{run_nr}"
            / "output"
            / f"{gen_nr}.generation_checkpoint"
        )

        if not generation_checkpoint_path.exists():
            print(f"Skipping {generation_checkpoint_path}, file not found")
            continue

        genotypes = load_genotypes(generation_checkpoint_path)

        for genotype in genotypes:
            column_names = extract_column_names(
                column_names,
                genotype,
            )
            row = make_df_row(run_nr, gen_nr, genotype)
            df_rows.append(row)

    df = pd.DataFrame(data=df_rows, columns=column_names)
    df.to_csv(persistent_dir / "fitnesses.csv", index=False)


if __name__ == "__main__":
    main()
