import functools
import pathlib
import typing
import uuid

import gge.experiments.settings
import gge.grammars.structured_grammatical_evolution as sge
import gge.paths
from gge.evolutionary.generations import EvaluatedGenotype, GenerationCheckpoint


def population_contains_genotype(
    population: typing.Iterable[EvaluatedGenotype],
    genotype_uuid: uuid.UUID,
) -> bool:
    matches = (ind.genotype.unique_id == genotype_uuid for ind in population)
    return any(matches)


def generation_checkpoint_contains_genotype(
    generation_checkpoint_path: pathlib.Path,
    genotype_uuid: uuid.UUID,
) -> bool:
    checkpoint = GenerationCheckpoint.load(generation_checkpoint_path)
    return population_contains_genotype(
        checkpoint.get_population(),
        genotype_uuid,
    )


@functools.cache
def find_generation_checkpoint(
    genotype_uuid: uuid.UUID,
    base_search_dir: pathlib.Path,
) -> pathlib.Path:

    checkpoint_paths = list(
        base_search_dir.rglob(f"*{gge.paths.GENERATION_CHECKPOINT_EXTENSION}")
    )

    if len(checkpoint_paths) == 0:
        raise ValueError("base_search_dir contains no generation checkpoint files")

    for path in checkpoint_paths:
        if generation_checkpoint_contains_genotype(path, genotype_uuid):
            return path

    raise ValueError("file not found")


def find_and_load_genotype(
    genotype_uuid: uuid.UUID,
    base_search_dir: pathlib.Path,
) -> gge.evolutionary.generations.EvaluatedGenotype:
    path = find_generation_checkpoint(genotype_uuid, base_search_dir)
    generation = GenerationCheckpoint.load(path)
    for ev in generation.get_population():
        if ev.genotype.unique_id == genotype_uuid:
            return ev

    raise ValueError("this should never happen")


def get_settings_path(
    genotype_uuid: uuid.UUID,
    base_search_dir: pathlib.Path,
) -> pathlib.Path:
    generation_checkpoint_path = find_generation_checkpoint(
        genotype_uuid, base_search_dir
    )
    expected_settings_path = generation_checkpoint_path.parent.parent / "settings.yaml"
    if expected_settings_path.is_file():
        return expected_settings_path

    raise ValueError(
        f"settings file is not present at the expected path=<{expected_settings_path}>"
    )


def main() -> None:
    genotype_uuid = uuid.UUID("7a8c3b21152b4b30b1d62bf48ca77d5e")

    base_search_dir = (
        pathlib.Path("/workspaces")
        / "gge"
        / "gge"
        / "playground"
        / "gitignored"
        / "cifar10"
    )

    settings_path = get_settings_path(
        genotype_uuid,
        base_search_dir,
    )

    evaluated_genotype = find_and_load_genotype(genotype_uuid, base_search_dir)
    settings = gge.experiments.settings.load_gge_settings(settings_path)

    tokenstream = sge.map_to_tokenstream(
        evaluated_genotype.genotype.backbone_genotype,
        settings.grammar,
    )

    print(tokenstream)


if __name__ == "__main__":
    main()
