#!/bin/env python

import os  # noqa
import pathlib
import shutil
import subprocess
import sys
import tempfile
import typing

import tomli
import typer

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


@app.command()
def init_create(settings_path: pathlib.Path = SETTINGS_OPTION) -> None:
    with settings_path.open("rb") as file:
        settings = tomli.load(file)

    import pprint

    pprint.pprint(settings)


@app.command()
def run_evolution() -> None:
    print("evolved")


if __name__ == "__main__":
    app()
