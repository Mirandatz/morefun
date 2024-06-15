#!/bin/env python

import os
from pathlib import Path
from subprocess import check_call
from typing import Annotated

import typer
import yaml
from paths import find_repository_root


def main(
    settings_path: Annotated[
        Path,
        typer.Option(
            "-s",
            "--settings-path",
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
    settings_yaml = yaml.safe_load(settings_path.read_text())

    container_code_dir = Path("/app/code")
    container_settings_path = Path("/app/settings.yaml")
    container_dataset_dir = Path(settings_yaml["dataset"]["partitions_dir"])
    container_output_dir = Path(settings_yaml["output"]["directory"])

    # container_cmd = "ls"
    container_cmd = f"python -m morefun.experiments.v2.initialize_population -s {container_settings_path}"

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
        f"-v={find_repository_root()}:{container_code_dir}:ro",
        f"-v={dataset_dir}:{container_dataset_dir}:ro",
        f"-v={settings_path}:{container_settings_path}:ro",
        f"-v={output_dir}:{container_output_dir}",
        f"--workdir={container_code_dir}",
        "mirandatz/morefun:dev_env",
        "bash",
        "-c",
        container_cmd,
    )

    check_call(args)

    return 0


if __name__ == "__main__":
    typer.run(main)
