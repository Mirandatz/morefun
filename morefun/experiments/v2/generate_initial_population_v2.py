#!/bin/env python

from pathlib import Path
from typing import Annotated

import typer

import morefun.evolutionary.fitnesses as gf
import morefun.evolutionary.generations
import morefun.evolutionary.novelty
import morefun.experiments.create_initial_population_genotypes as mf_init
import morefun.experiments.v2.configs as mf_cfg
import morefun.paths
import morefun.phenotypes
import morefun.randomness


def main(
    config_path: Annotated[
        Path,
        typer.Option(
            "-c",
            "--config-path",
            file_okay=True,
            exists=True,
            readable=True,
            dir_okay=False,
        ),
    ],
) -> None:
    config_path = mf_cfg.load_morefun_settings(config_path)

    mf_cfg.configure_logger(config_path.output)
    mf_cfg.configure_tensorflow(config_path.tensorflow)

    rng_seed = config_path.experiment.rng_seed

    individuals = mf_init.create_initial_population(
        pop_size=config_path.initialization.population_size,
        grammar=config_path.grammar,
        filter=config_path.initialization.individual_filter,
        rng_seed=rng_seed,
    )

    metrics = mf_cfg.make_metrics(
        dataset=config_path.dataset,
        fitness=config_path.evolution.fitness_settings,
        output=config_path.output,
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
        config_path.output.directory, generation_number
    )

    checkpoint.save(save_path)


if __name__ == "__main__":
    typer.run(main)
