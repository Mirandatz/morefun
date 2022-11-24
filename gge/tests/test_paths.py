import pathlib
import tempfile

from hypothesis import given
from hypothesis import strategies as hs

import gge.paths


def test_grammars_dir() -> None:
    path = gge.paths.get_grammars_dir()
    assert (path / "upper_grammar.lark").is_file()
    assert (path / "lower_grammar.lark").is_file()
    assert (path / "terminals.lark").is_file()


@given(generation_nr=hs.integers(min_value=0, max_value=99))
def test_get_generation_checkpoint_path(generation_nr: int) -> None:
    base_output_dir = pathlib.Path("/dev/shm")
    expected = (base_output_dir / str(generation_nr)).with_suffix(
        ".generation_checkpoint"
    )
    actual = gge.paths.get_generation_checkpoint_path(base_output_dir, generation_nr)
    assert expected == actual


@given(
    generation_numbers=hs.lists(
        hs.integers(
            min_value=0,
            max_value=99,
        ),
        min_size=1,
        max_size=99,
    )
)
def test_get_latest_generation_output_path(generation_numbers: list[int]) -> None:
    with tempfile.TemporaryDirectory(dir="/dev/shm") as _temp_dir:
        temp_dir = pathlib.Path(_temp_dir)

        for gen_nr in generation_numbers:
            gge.paths.get_generation_checkpoint_path(temp_dir, gen_nr).write_text("")

        expected = gge.paths.get_generation_checkpoint_path(
            temp_dir, max(generation_numbers)
        )
        actual = gge.paths.get_latest_generation_checkpoint_path(temp_dir)
        assert expected == actual
