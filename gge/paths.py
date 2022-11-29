import functools
import pathlib
import uuid

GENERATION_CHECKPOINT_EXTENSION = ".generation_checkpoint"
PHENOTYPE_EXTENSION = ".phenotype"


@functools.cache
def get_project_root_dir() -> pathlib.Path:
    this_file = pathlib.Path(__file__)
    filesystem_root = pathlib.Path(this_file.root)

    current_dir = this_file.parent

    while current_dir != filesystem_root:
        if (current_dir / ".gge_root").exists():
            return current_dir

        current_dir = current_dir.parent

    raise ValueError("unable to find `project root directory`")


def get_grammars_dir() -> pathlib.Path:
    return get_project_root_dir() / "gge" / "grammars" / "files"


def get_generation_checkpoint_path(
    output_dir: pathlib.Path, generation_number: int
) -> pathlib.Path:
    assert generation_number >= 0
    return (output_dir / str(generation_number)).with_suffix(
        GENERATION_CHECKPOINT_EXTENSION
    )


def get_latest_generation_checkpoint_path(search_dir: pathlib.Path) -> pathlib.Path:
    if not search_dir.is_dir():
        raise ValueError("search_dir is not a directory")

    candidates = list(search_dir.glob(f"*{GENERATION_CHECKPOINT_EXTENSION}"))
    if not candidates:
        raise ValueError("search_dir does not contain generation-output files")

    return max(candidates, key=lambda path: int(path.stem))


def get_phenotype_path(output_dir: pathlib.Path, uuid: uuid.UUID) -> pathlib.Path:
    return (output_dir / uuid.hex).with_suffix(PHENOTYPE_EXTENSION)


def get_keras_model_path(output_dir: pathlib.Path, uuid: uuid.UUID) -> pathlib.Path:
    return output_dir / f"{uuid.hex} _keras_model"
