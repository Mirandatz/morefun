#!/bin/env python

from pathlib import Path
from typing import Annotated

import typer

from morefun.evolutionary.generations import (
    GenerationCheckpoint,
    run_multiple_generations,
)
from morefun.experiments.v2.settings import (
    configure_logger,
    configure_tensorflow,
    load_morefun_settings,
    make_metrics,
    make_mutation_params,
)
from morefun.paths import get_latest_generation_checkpoint_path


def main(
    settings_path: Annotated[
        Path,
        typer.Option(
            "-s",
            "--settings-path",
            file_okay=True,
            exists=True,
            readable=True,
            dir_okay=False,
        ),
    ],
    generations: Annotated[int, typer.Option("--generations", min=1)],
) -> None:
    settings = load_morefun_settings(settings_path)
    configure_logger(settings.output)
    configure_tensorflow(settings.tensorflow)

    latest_checkpoint = GenerationCheckpoint.load(
        get_latest_generation_checkpoint_path(settings.output.directory)
    )

    current_generation_number = latest_checkpoint.get_generation_number() + 1

    mutation_params = make_mutation_params(
        mutation=settings.evolution.mutation_settings,
        grammar=settings.grammar,
    )

    metrics = make_metrics(
        dataset=settings.dataset,
        fitness=settings.evolution.fitness_settings,
        output=settings.output,
    )

    run_multiple_generations(
        starting_generation_number=current_generation_number,
        number_of_generations_to_run=generations,
        initial_population=latest_checkpoint.get_population(),
        grammar=settings.grammar,
        mutation_params=mutation_params,
        metrics=metrics,
        novelty_tracker=latest_checkpoint.get_novelty_tracker(),
        rng=latest_checkpoint.get_rng(),
        output_dir=settings.output.directory,
    )


if __name__ == "__main__":
    typer.run(main)
