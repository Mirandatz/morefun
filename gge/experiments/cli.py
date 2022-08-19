import pathlib

import typer

import gge.experiments.create_initial_population_genotypes as exp_init_create
import gge.experiments.settings as gset
import gge.fitnesses as gf
import gge.persistence as gp

SETTINGS_OPTION = typer.Option(
    pathlib.Path("/gge/settings.toml"),
    "--settings-path",
    file_okay=True,
    exists=True,
    readable=True,
    dir_okay=False,
)


app = typer.Typer()


@app.command(name="init-create")
def create_initial_population(
    settings_path: pathlib.Path = SETTINGS_OPTION,
) -> None:
    settings = gset.load_settings(settings_path)

    gset.configure_logger(settings)

    base_output_dir = gset.get_base_output_dir(settings)

    grammar = gset.get_grammar(settings)

    filter = exp_init_create.IndividualFilter(
        max_network_depth=settings["initialization"]["max_network_depth"],
        max_wide_layers=settings["initialization"]["max_wide_layers"],
        max_layer_width=settings["initialization"]["wide_layer_threshold"],
        max_network_params=settings["initialization"]["max_network_params"],
    )

    population = exp_init_create.create_initial_population(
        pop_size=settings["initialization"]["population_size"],
        grammar=grammar,
        filter=filter,
        rng_seed=settings["experiment"]["rng_seed"],
    )

    genotypes = [indi.genotype for indi in population]

    # generations are 0-indexed, so first gen == 0
    generation = 0
    gp.save_generation_genotypes(genotypes, generation, base_output_dir)


@app.command(name="init-evaluate")
def evaluate_initial_population(
    settings_path: pathlib.Path = SETTINGS_OPTION,
) -> None:
    settings = gset.load_settings_and_configure_logger(settings_path)

    # generations are 0-indexed, so first gen == 0
    generation = 0

    base_output_dir = gset.get_base_output_dir(settings)
    genotypes_dir = gp.get_genotypes_dir(generation, base_output_dir)
    genotypes = [gp.load_genotype(path) for path in genotypes_dir.iterdir()]

    if len(genotypes) == 0:
        raise ValueError(f"the genotypes directory is empty, path=<{genotypes_dir}>")

    fitness_evaluation_params = gset.get_fitness_evaluation_params(settings)
    fers = [gf.evaluate(g, fitness_evaluation_params) for g in genotypes]

    gp.save_generation_fitness_evaluation_results(fers, generation, base_output_dir)


def get_newest_generation_dir(generations_dir: pathlib.Path) -> pathlib.Path:
    if not generations_dir.is_dir():
        raise ValueError(
            f"`generations_dir` is not a directorty, path=<{generations_dir}>"
        )

    subdirs = [p for p in generations_dir.iterdir() if p.is_dir()]

    if len(subdirs) == 0:
        raise ValueError(f"`generations_dir` is empty, path=<{generations_dir}>")

    return max(subdirs, key=lambda path: int(path.name))


@app.command(name="evolve")
def evolutionary_loop(
    settings_path: pathlib.Path = SETTINGS_OPTION,
    generations: int = typer.Option(..., "--generations", min=1),
) -> None:
    raise NotImplementedError()
    # settings = gset.load_settings_and_configure_logger(settings_path)

    # newest_generation = get_newest_generation_dir(
    #     generations_dir=gset.get_generations_dir(settings)
    # )

    # fitness_evaluation_results = gge.persistence.population_fitnesses(newest_generation)

    # fitness_evaluation_params = gset.get_fitness_evaluation_params(settings)

    # output_dir = gset.get_initial_population_fitness_dir(settings)
    # output_dir.mkdir(parents=True, exist_ok=True)

    # exp_eval.evaluate_population(
    #     genotypes,
    #     fitness_evaluation_params,
    #     output_dir=output_dir,
    # )


if __name__ == "__main__":
    app()
