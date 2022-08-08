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


def create_gge_tgz(repository_path: pathlib.Path, tgz_path: pathlib.Path) -> None:
    assert repository_path.is_dir()

    proc = subprocess.run(
        args=["git", "archive", "--format=tgz", "HEAD"],
        cwd=repository_path,
        capture_output=True,
        check=True,
    )

    tgz_path.write_bytes(proc.stdout)


def docker_run(host_mount_point: pathlib.Path, command: str) -> None:
    subprocess.run(
        args=[
            "docker",
            "run",
            f"--user={os.getuid()}:{os.getgid()}",
            "--rm",
            "--runtime=nvidia",
            "--workdir=/gge",
            f"-v={host_mount_point}:/gge",
            DOCKER_IMAGE_NAME,
            "bash",
            "-e",
            "-u",
            "-x",
            "-o",
            "pipefail",
            "-c",
            command,
        ],
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

    with tempfile.TemporaryDirectory(
        prefix="gge_workdir_",
        dir="/dev/shm",
    ) as workdir_str:
        workdir = pathlib.Path(workdir_str)

        shutil.copy(
            src=settings_path,
            dst=workdir / "settings.toml",
        )

        create_gge_tgz(
            repository_path=get_first_valid_host_path(settings["gge"]["host_path"]),
            tgz_path=workdir / "gge.tgz",
        )

        docker_run(
            host_mount_point=workdir,
            command="mkdir -p gge \
                    && tar -zxf /gge/gge.tgz -C gge \
                    && cd /gge/gge \
                    && pyenv local --unset \
                    && ls gge/experiments",
        )

        # shutil.copy(
        #     src=workdir / "generations" / "0" / "genotypes",
        #     dst=settings_path.parent / "generations" / "0" / "genotypes",
        # )


@app.command()
def run_evolution() -> None:
    print("evolved")


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent / "tmp" / "settings.toml"
    create_initial_population(path)
    # app()
