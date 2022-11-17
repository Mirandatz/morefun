import pathlib

import typer

import gge.evolutionary.fitnesses as gf
import gge.evolutionary.generations
import gge.experiments.create_initial_population_genotypes as gge_init
import gge.experiments.settings as gset
import gge.novelty
import gge.persistence
import gge.phenotypes
import gge.randomness

SETTINGS_OPTION = typer.Option(
    pathlib.Path("/gge/settings.toml"),
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
    settings = gset.load_gge_settings(settings_path)
    gset.configure_logger(settings.output)
    gset.configure_tensorflow(settings.tensorflow)

    rng_seed = settings.experiment.rng_seed

    individuals = gge_init.create_initial_population(
        pop_size=settings.initialization.population_size,
        grammar=settings.grammar,
        filter=settings.initialization.individual_filter,
        rng_seed=rng_seed,
    )

    metrics = gset.make_metrics(
        dataset=settings.dataset,
        fitness=settings.evolution.fitness_settings,
    )

    genotypes = [ind.genotype for ind in individuals]
    phenotypes = [ind.phenotype for ind in individuals]
    fitnesses = {
        ind.genotype: gf.evaluate(ind.phenotype, metrics) for ind in individuals
    }

    known_genotypes = set(genotypes)
    known_phenotypes = set(phenotypes)
    novelty_tracker = gge.novelty.NoveltyTracker(
        known_genotypes=known_genotypes,
        known_phenotypes=known_phenotypes,
    )

    # generations are 0-indexed, so first gen == 0
    gge.persistence.save_generational_artifacts(
        generation_number=0,
        fittest=fitnesses,
        novelty_tracker=novelty_tracker,
        rng=gge.randomness.create_rng(rng_seed),
        output_dir=settings.output.directory,
    )


@app.command(name="evolve")
def evolutionary_loop(
    settings_path: pathlib.Path = SETTINGS_OPTION,
    generations: int = typer.Option(..., "--generations", min=1),
) -> None:
    settings = gset.load_gge_settings(settings_path)
    gset.configure_logger(settings.output)
    gset.configure_tensorflow(settings.tensorflow)

    latest_gen_output = gge.persistence.load_latest_generational_artifacts(
        settings.output.directory
    )

    current_generation_number = latest_gen_output.get_generation_number() + 1

    mutation_params = gset.make_mutation_params(
        mutation=settings.evolution.mutation_settings,
        grammar=settings.grammar,
    )

    metrics = gset.make_metrics(
        dataset=settings.dataset,
        fitness=settings.evolution.fitness_settings,
    )

    gge.evolutionary.generations.run_multiple_generations(
        starting_generation_number=current_generation_number,
        number_of_generations_to_run=generations,
        initial_population=latest_gen_output.get_fittest(),
        grammar=settings.grammar,
        mutation_params=mutation_params,
        metrics=metrics,
        novelty_tracker=latest_gen_output.get_novelty_tracker(),
        rng=latest_gen_output.get_rng(),
        output_dir=settings.output.directory,
    )


if __name__ == "__main__":
    app()
