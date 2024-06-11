#!/bin/env python

import os
from pathlib import Path
from subprocess import check_call
from typing import Annotated

import typer
from paths import find_repository_root


def main(
    config_path: Annotated[
        Path,
        typer.Option(
            "-c",
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    dataset_dir: Annotated[
        Path,
        typer.Option(
            "-d",
            "--dataset",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    repo_root = find_repository_root()

    output_dir.mkdir(parents=True, exist_ok=True)

    uid = os.getuid()
    gid = os.getgid()

    args = (
        "docker",
        "run",
        "--rm",
        "--runtime=nvidia",
        "--shm-size=8gb",
        f"--user={uid}:{gid}",
        f"-v={repo_root}:/app/code:ro",
        f"-v={dataset_dir}:/app/dataset:ro",
        f"-v={config_path}:/app/settings.yaml:ro",
        f"-v={output_dir}:/app/output",
        "--workdir=/app/code",
        "mirandatz/morefun:dev_env",
        "bash",
        "-c",
        "ls /app/dataset",
        # "source /app/.venv/bin/activate && python -m morefun.experiments.cli $*",
    )

    print(args)
    check_call(args)


if __name__ == "__main__":
    typer.run(main)
