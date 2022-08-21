import pathlib
import tempfile

import hypothesis.strategies as hs
from hypothesis import given

import gge.persistence as gp


def test_generations_dir() -> None:
    """
    Can create the "base-generations-dir".
    """

    with tempfile.TemporaryDirectory(dir="/dev/shm") as _output_dir:
        base_output_dir = pathlib.Path(_output_dir)

        actual = gp.get_generations_dir(base_output_dir)
        expected = base_output_dir / "generations"

        assert expected == actual


@given(generation=hs.integers(min_value=0, max_value=99))
def test_generation_dir(
    generation: int,
) -> None:
    """
    Can create the directory of a specific generation.
    """

    with tempfile.TemporaryDirectory(dir="/dev/shm") as _output_dir:
        base_output_dir = pathlib.Path(_output_dir)

        actual = gp.get_generation_dir(generation, base_output_dir)
        expected = base_output_dir / "generations" / str(generation)
        assert expected == actual


@given(generation=hs.integers(min_value=0, max_value=99))
def test_genotypes_dir(
    generation: int,
) -> None:
    """
    Can create the directory for genotypes of a given generation.
    """

    with tempfile.TemporaryDirectory(dir="/dev/shm") as _output_dir:
        base_output_dir = pathlib.Path(_output_dir)

        actual = gp.get_genotypes_dir(generation, base_output_dir)
        expected = base_output_dir / "generations" / str(generation) / "genotypes"
        assert expected == actual


@given(generation=hs.integers(min_value=0, max_value=99))
def test_fitness_evaluation_results_dir(
    generation: int,
) -> None:
    """
    Can create the directory for fitness evaluation results of a given generation.
    """

    with tempfile.TemporaryDirectory(dir="/dev/shm") as _output_dir:
        base_output_dir = pathlib.Path(_output_dir)

        actual = gp.get_fitness_evaluation_results_dir(generation, base_output_dir)
        expected = (
            base_output_dir
            / "generations"
            / str(generation)
            / "fitness_evaluation_results"
        )
        assert expected == actual
