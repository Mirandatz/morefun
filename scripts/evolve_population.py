#!/bin/env python

from pathlib import Path
from subprocess import check_call
from typing import Annotated

import typer
from containerization import extract_mount_points_from_settings, make_docker_base_args


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
    mount_points = extract_mount_points_from_settings(
        dataset_dir=dataset_dir,
        settings_path=settings_path,
        output_path=output_dir,
    )
    base_docker_args = make_docker_base_args(mount_points)
    subprocess_args = base_docker_args + [
        "bash",
        "-c",
        f"python -m morefun.experiments.v2.evolve_population -s {mount_points.settings.container_path}",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    check_call(subprocess_args)
    return 0


if __name__ == "__main__":
    typer.run(main)
