import pathlib

import typer

import gge.evolution
import gge.experiments.create_initial_population_genotypes as gge_init
import gge.experiments.settings as gset
import gge.fitnesses as gf
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

    rng_seed = settings.experiment.rng_seed

    individuals = gge_init.create_initial_population(
        pop_size=settings.initialization.population_size,
        grammar=settings.grammar,
        filter=settings.initialization.individual_filter,
        rng_seed=rng_seed,
    )

    genotypes = [ind.genotype for ind in individuals]
    phenotypes = [ind.phenotype for ind in individuals]

    fitness_evaluation_params = gset.make_fitness_evaluation_params(
        dataset=settings.dataset,
        fitness=settings.evolution.fitness_settings,
        grammar=settings.grammar,
    )

    fitness_evaluation_results = [
        gf.evaluate(g, fitness_evaluation_params) for g in genotypes
    ]

    known_genotypes = set(genotypes)
    known_phenotypes = set(phenotypes)
    novelty_tracker = gge.novelty.NoveltyTracker(
        known_genotypes=known_genotypes,
        known_phenotypes=known_phenotypes,
    )

    # generations are 0-indexed, so first gen == 0
    gge.persistence.save_generation_output(
        generation_number=0,
        fittest=fitness_evaluation_results,
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

    latest_gen_output = gge.persistence.load_latest_generation_output(
        settings.output.directory
    )

    current_generation_number = latest_gen_output.generation_number + 1

    mutation_params = gset.make_mutation_params(
        mutation=settings.evolution.mutation_settings,
        grammar=settings.grammar,
    )

    fitness_params = gset.make_fitness_evaluation_params(
        dataset=settings.dataset,
        fitness=settings.evolution.fitness_settings,
        grammar=settings.grammar,
    )

    gge.evolution.run_evolutionary_loop(
        starting_generation_number=current_generation_number,
        number_of_generations_to_run=generations,
        initial_population=latest_gen_output.fittest,
        mutation_params=mutation_params,
        fitness_params=fitness_params,
        novelty_tracker=latest_gen_output.novelty_tracker,
        rng=latest_gen_output.rng,
        output_dir=settings.output.directory,
    )


@app.command(name="final-train")
def final_train(
    settings_path: pathlib.Path = SETTINGS_OPTION,
) -> None:
    raise NotImplementedError()
    # settings = gset.load_settings_and_configure_logger(settings_path)

    # base_output_dir = gset.get_base_output_dir(settings)
    # latest_gen_output = gge.persistence.load_latest_generation_output(base_output_dir)

    # best_of_the_best = max(
    #     latest_gen_output.fittest,
    #     key=lambda fer: gf.get_effective_fitness(fer),
    # )

    # final_train_fitness_params = gset.get_final_performance_evaluation_params(settings)


if __name__ == "__main__":
    app()
