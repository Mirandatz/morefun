#!/bin/env python

from pathlib import Path
from typing import Annotated

import typer

import morefun.evolutionary.fitnesses as gf
import morefun.evolutionary.generations
import morefun.evolutionary.novelty
import morefun.experiments.create_initial_population_genotypes as mf_init
import morefun.experiments.settings as mfs
import morefun.paths
import morefun.phenotypes
import morefun.randomness


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
) -> None:
    settings = mfs.load_morefun_settings(settings_path)

    mfs.configure_logger(settings.output)
    mfs.configure_tensorflow(settings.tensorflow)

    rng_seed = settings.experiment.rng_seed

    individuals = mf_init.create_initial_population(
        pop_size=settings.initialization.population_size,
        grammar=settings.grammar,
        filter=settings.initialization.individual_filter,
        rng_seed=rng_seed,
    )

    metrics = mfs.make_metrics(
        dataset=settings.dataset,
        fitness=settings.evolution.fitness_settings,
        output=settings.output,
    )

    genotypes = [ind.genotype for ind in individuals]
    phenotypes = [ind.phenotype for ind in individuals]
    fitnesses = [gf.evaluate(ind.phenotype, metrics) for ind in individuals]

    known_genotypes = set(genotypes)
    known_phenotypes = set(phenotypes)
    novelty_tracker = morefun.evolutionary.novelty.NoveltyTracker(
        known_genotypes=known_genotypes,
        known_phenotypes=known_phenotypes,
    )

    initial_population = [
        morefun.evolutionary.generations.EvaluatedGenotype(g, p, f)
        for g, p, f in zip(
            genotypes,
            phenotypes,
            fitnesses,
        )
    ]

    # generations are 0-indexed, so first gen == 0
    generation_number = 0

    checkpoint = morefun.evolutionary.generations.GenerationCheckpoint(
        generation_number=generation_number,
        population=tuple(initial_population),
        rng=morefun.randomness.create_rng(rng_seed),
        novelty_tracker=novelty_tracker,
    )

    save_path = morefun.paths.get_generation_checkpoint_path(
        settings.output.directory, generation_number
    )

    checkpoint.save(save_path)


if __name__ == "__main__":
    typer.run(main)
