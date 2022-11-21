import functools
import pathlib

GENERATION_OUTPUT_EXTENSION = ".gen_out2"


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


def get_generation_output_path(
    output_dir: pathlib.Path,
    generation_nr: int,
) -> pathlib.Path:
    assert generation_nr >= 0

    return (output_dir / str(generation_nr)).with_suffix(GENERATION_OUTPUT_EXTENSION)


def get_latest_generation_output_path(search_dir: pathlib.Path) -> pathlib.Path:
    if not search_dir.is_dir():
        raise ValueError("search_dir is not a directory")

    candidates = list(search_dir.glob(f"*{GENERATION_OUTPUT_EXTENSION}"))
    if not candidates:
        raise ValueError("search_dir does not contain generation-output files")

    return max(candidates, key=lambda path: int(path.stem))
