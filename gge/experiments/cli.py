#!/bin/env python

import os  # noqa
import pathlib
import shutil
import subprocess
import sys  # noqa
import tempfile  # noqa
import typing

import docker  # noqa
import tomli
import typer

# from loguru import logger  # noqa

Settings = dict[str, typing.Any]

app = typer.Typer()

DOCKER_IMAGE_NAME = "mirandatz/gge:dev_env"


def get_first_valid_host_path(host_paths: list[str]) -> pathlib.Path:
    as_pathlib_paths = map(pathlib.Path, host_paths)
    for path in as_pathlib_paths:
        if path.exists():
            return path

    raise ValueError("no path in `host_path` exists")


def copy_gge(repository_path: pathlib.Path, output_path: pathlib.Path) -> None:
    assert repository_path.is_dir()

    proc = subprocess.run(
        args=["git", "archive", "--format=tgz", "HEAD"],
        cwd=repository_path,
        capture_output=True,
        check=True,
    )

    output_path.mkdir(exist_ok=True, parents=True)

    subprocess.run(
        args=["tar", "-zxf", "-"],
        cwd=output_path,
        input=proc.stdout,
        check=True,
    )


def docker_run(host_mount_point: pathlib.Path, commands: list[str]) -> None:
    args = [
        "docker",
        "run",
        f"--user={os.getuid()}:{os.getgid()}",
        "--rm",
        "--runtime=nvidia",
        "--workdir=/gge",
        f"-v={host_mount_point}:/gge",
        DOCKER_IMAGE_NAME,
        *commands,
    ]

    subprocess.run(
        args=args,
        check=True,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


@app.command()
def create_initial_population(
    settings_path: pathlib.Path = typer.Option(..., "--settings-file")
) -> None:
    with settings_path.open("rb") as file:
        settings = tomli.load(file)

    with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdir:
        mount_point = pathlib.Path(tmpdir)

        shutil.copy(
            src=settings_path,
            dst=mount_point / "settings.toml",
        )

        copy_gge(
            repository_path=get_first_valid_host_path(settings["gge"]["host_path"]),
            output_path=mount_point / "gge",
        )

        docker_run(
            host_mount_point=mount_point,
            commands=[
                "bash",
                "-euo",
                "pipefail",
                "-c",
                "source /venv/bin/activate && python -m gge.experiments.create_initial_population",
            ],
        )

        # shutil.copy(
        #     src=workdir / "generations" / "0" / "genotypes",
        #     dst=settings_path.parent / "generations" / "0" / "genotypes",
        # )


@app.command()
def run_evolution() -> None:
    print("evolved")


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent / "cifar10" / "settings.toml"
    create_initial_population(path)
    # app()
