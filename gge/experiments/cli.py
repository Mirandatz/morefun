import os  # noqa
import pathlib
import shutil  # noqa
import subprocess  # noqa
import sys  # noqa
import tempfile  # noqa
import typing  # noqa

import tomli  # noqa
import typer

import gge.experiments.create_initial_population_genotypes as exp_init_create
import gge.experiments.evaluate_genotypes as exp_eval
import gge.experiments.settings as gset
import gge.persistence

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

    output_dir = gset.get_initial_population_genotypes_dir(settings)
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

    gge.persistence.save_population_genotypes(genotypes, output_dir)


@app.command(name="init-evaluate")
def evaluate_initial_population(
    settings_path: pathlib.Path = SETTINGS_OPTION,
) -> None:
    settings = gset.load_settings_and_configure_logger(settings_path)

    genotypes = gge.persistence.load_population_genotypes(
        gset.get_initial_population_genotypes_dir(settings)
    )

    fitness_evaluation_params = gset.get_fitness_evaluation_params(settings)

    output_dir = gset.get_initial_population_fitness_dir(settings)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_eval.evaluate_population(
        genotypes,
        fitness_evaluation_params,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    app()
