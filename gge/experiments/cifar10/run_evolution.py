import pathlib

import typer
from loguru import logger


def validate_output_dir(path: pathlib.Path) -> None:
    if not path.exists():
        logger.info(f"output dir does not exist, creating. <{path=}>")

        try:
            path.mkdir()
            return
        except Exception as ex:
            logger.error(f"unable to create output dir, <{ex=}>")
            exit(-1)

    else:
        logger.info(f"output dir already exists, checking if empty. <{path=}>")
        for file in path.iterdir():
            logger.error(f"output dir is not empty, contains <{file=}>")
            exit(-1)


def main(
    dataset_dir: pathlib.Path = typer.Option(
        ...,
        "-d",
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "-o",
        file_okay=False,
        dir_okay=True,
    ),
) -> None:

    validate_output_dir(output_dir)


if __name__ == "__main__":
    typer.run(main)
