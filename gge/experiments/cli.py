import os  # noqa
import pathlib
import shutil
import subprocess
import sys
import tempfile
import typing

import tomli
import typer

import gge.experiments.create_initial_population_genotypes as exp_init_create
import gge.grammars as gr

SETTINGS_OPTION = typer.Option(
    pathlib.Path("/gge/settings.toml"),
    "--settings-path",
    file_okay=True,
    exists=True,
    readable=True,
    dir_okay=False,
)

Settings = dict[str, typing.Any]


app = typer.Typer()


def get_first_valid_host_path(host_paths: list[str]) -> pathlib.Path:
    as_pathlib_paths = map(pathlib.Path, host_paths)
    for path in as_pathlib_paths:
        if path.exists():
            return path

    raise ValueError("no path in `host_path` exists")


@app.command(name="init-create")
def create_initial_population_genotypes(
    settings_path: pathlib.Path = SETTINGS_OPTION,
) -> None:
    with settings_path.open("rb") as file:
        settings = tomli.load(file)

    output_dir = pathlib.Path(settings["initialization"]["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    grammar = gr.Grammar(settings["grammar"]["raw"])

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

    exp_init_create.save_population(population, output_dir)


@app.command()
def run_evolution() -> None:
    print("evolved")


if __name__ == "__main__":
    app()
