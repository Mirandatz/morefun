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
    settings = gset.load_settings(settings_path)

    gset.configure_logger(settings)

    base_output_dir = gset.get_base_output_dir(settings)

    grammar = gset.get_grammar()
    rng_seed = gset.get_rng_seed()

    genotypes = gge_init.create_initial_population(
        pop_size=gset.get_initial_population_size(settings),
        grammar=grammar,
        filter=gset.get_initialization_individual_filter(settings),
        rng_seed=rng_seed,
    )

    fitness_evaluation_params = gset.get_fitness_evaluation_params(settings)
    fitness_evaluation_results = [
        gf.evaluate(g, fitness_evaluation_params) for g in genotypes
    ]

    known_genotypes = set(genotypes)
    known_phenotypes = set(gge.phenotypes.translate(g, grammar) for g in set(genotypes))
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
        output_dir=base_output_dir,
    )


@app.command(name="evolve")
def evolutionary_loop(
    settings_path: pathlib.Path = SETTINGS_OPTION,
    generations: int = typer.Option(..., "--generations", min=1),
) -> None:
    settings = gset.load_settings_and_configure_logger(settings_path)

    base_output_dir = gset.get_base_output_dir(settings)

    latest_gen_output = gge.persistence.load_latest_generation_output(base_output_dir)
    current_generation_number = latest_gen_output.generation_number + 1
    mutation_params = gset.get_mutation_params(settings)
    fitness_params = gset.get_fitness_evaluation_params(settings)

    gge.evolution.run_evolutionary_loop(
        starting_generation_number=current_generation_number,
        number_of_generations_to_run=generations,
        initial_population=latest_gen_output.fittest,
        mutation_params=mutation_params,
        fitness_params=fitness_params,
        novelty_tracker=latest_gen_output.novelty_tracker,
        rng=latest_gen_output.rng,
        output_dir=base_output_dir,
    )


if __name__ == "__main__":
    app()
