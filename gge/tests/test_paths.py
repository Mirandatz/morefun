import pathlib

from hypothesis import given
from hypothesis import strategies as hs

import gge.paths


def test_grammars_dir() -> None:
    path = gge.paths.get_grammars_dir()
    assert (path / "upper_grammar.lark").is_file()
    assert (path / "lower_grammar.lark").is_file()
    assert (path / "terminals.lark").is_file()


@given(generation_nr=hs.integers(min_value=0, max_value=99))
def test_get_generation_output_path(generation_nr: int) -> None:
    base_output_dir = pathlib.Path("/dev/shm")
    expected = (base_output_dir / str(generation_nr)).with_suffix(".gen_out2")
    actual = gge.paths.get_generation_output_path(base_output_dir, generation_nr)
    assert expected == actual
