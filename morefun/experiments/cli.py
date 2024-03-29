import pathlib

import typer
from loguru import logger

import morefun.evolutionary.fitnesses as gf
import morefun.evolutionary.generations
import morefun.evolutionary.novelty
import morefun.experiments.create_initial_population_genotypes as mf_init
import morefun.experiments.settings as mf_cfg
import morefun.paths
import morefun.phenotypes
import morefun.randomness

SETTINGS_OPTION = typer.Option(
    pathlib.Path("/app/settings.yaml"),
    "--settings-path",
    file_okay=True,
    exists=True,
    readable=True,
    dir_okay=False,
)


app = typer.Typer()


@app.command(name="initialize")
def create_and_evaluate_initial_population(
    settings_path: pathlib.Path = SETTINGS_OPTION,
) -> None:
    settings = mf_cfg.load_morefun_settings(settings_path)
    mf_cfg.configure_logger(settings.output)
    mf_cfg.configure_tensorflow(settings.tensorflow)

    rng_seed = settings.experiment.rng_seed

    individuals = mf_init.create_initial_population(
        pop_size=settings.initialization.population_size,
        grammar=settings.grammar,
        filter=settings.initialization.individual_filter,
        rng_seed=rng_seed,
    )

    metrics = mf_cfg.make_metrics(
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


@app.command(name="evolve")
def evolutionary_loop(
    settings_path: pathlib.Path = SETTINGS_OPTION,
    generations: int = typer.Option(..., "--generations", min=1),
) -> None:
    settings = mf_cfg.load_morefun_settings(settings_path)
    mf_cfg.configure_logger(settings.output)
    mf_cfg.configure_tensorflow(settings.tensorflow)

    latest_checkpoint = morefun.evolutionary.generations.GenerationCheckpoint.load(
        morefun.paths.get_latest_generation_checkpoint_path(settings.output.directory)
    )

    current_generation_number = latest_checkpoint.get_generation_number() + 1

    mutation_params = mf_cfg.make_mutation_params(
        mutation=settings.evolution.mutation_settings,
        grammar=settings.grammar,
    )

    metrics = mf_cfg.make_metrics(
        dataset=settings.dataset,
        fitness=settings.evolution.fitness_settings,
        output=settings.output,
    )

    morefun.evolutionary.generations.run_multiple_generations(
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


def export_models(settings: gset.GgeSettings) -> None:
    gset.configure_logger(settings.output)
    gset.configure_tensorflow(settings.tensorflow)

    logger.info(f"export architectures from directory=<{settings.output.directory}>")

    last_checkpoint_path = gge.paths.get_latest_generation_checkpoint_path(
        settings.output.directory
    )

    checkpoint = gge.evolutionary.generations.GenerationCheckpoint.load(
        last_checkpoint_path
    )

    for ev in checkpoint.get_population():
        logger.info(f"exporting model for genotype=<{ev.genotype.unique_id}>")

        model = gf.make_classification_model(
            ev.phenotype,
            input_shape=settings.dataset.input_shape,
            class_count=settings.dataset.class_count,
        )

        path = gge.paths.get_keras_model_path(
            settings.output.directory,
            ev.genotype.unique_id,
        )

        model.save(path)


@app.command(name="export-models")
def export_models_command(settings_path: pathlib.Path = SETTINGS_OPTION) -> int:
    settings = gset.load_gge_settings(settings_path)
    export_models(settings)

    return 0


if __name__ == "__main__":
    app()
