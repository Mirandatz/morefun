import pathlib
import pickle

from loguru import logger

import gge.composite_genotypes as cg
import gge.evolutionary.fitnesses as gf
import gge.novelty
import gge.randomness as rand

GENERATION_OUTPUT_EXTENSION = ".gen_out2"
TRAINING_HISTORY_EXTENSION = ".train_hist"
GENOTYPE_EXTENSION = ".genotype"
MODEL_EXTENSION = ".zipped_tf_model"
MODEL_EVALUATIONS_EXTENSION = ".csv"


class GenerationOutput:
    def __init__(
        self,
        generation_number: int,
        fittest: dict[cg.CompositeGenotype, gf.Fitness],
        novelty_tracker: gge.novelty.NoveltyTracker,
        rng: rand.RNG,
    ) -> None:
        self._generation_number = generation_number
        self._fittest = dict(fittest)
        self._novelty_tracker = novelty_tracker.copy()
        self._serialized_rng = pickle.dumps(rng, protocol=pickle.HIGHEST_PROTOCOL)

    def get_generation_number(self) -> int:
        return self._generation_number

    def get_fittest(self) -> dict[cg.CompositeGenotype, gf.Fitness]:
        return dict(self._fittest)

    def get_novelty_tracker(self) -> gge.novelty.NoveltyTracker:
        return self._novelty_tracker.copy()

    def get_rng(self) -> rand.RNG:
        rng = pickle.loads(self._serialized_rng)
        assert isinstance(rng, rand.RNG)
        return rng


def get_generational_artifacts_path(
    generation_number: int,
    directory: pathlib.Path,
) -> pathlib.Path:
    return (directory / str(generation_number)).with_suffix(GENERATION_OUTPUT_EXTENSION)


def get_genotype_path(
    genotype: cg.CompositeGenotype,
    dir: pathlib.Path,
) -> pathlib.Path:
    return (dir / genotype.unique_id.hex).with_suffix(GENOTYPE_EXTENSION)


def get_model_path(genotype_uuid_hex: str, dir: pathlib.Path) -> pathlib.Path:
    return (dir / genotype_uuid_hex).with_suffix(MODEL_EXTENSION)


def get_training_history_path(
    genotype_uuid_hex: str,
    dir: pathlib.Path,
) -> pathlib.Path:
    return (dir / genotype_uuid_hex).with_suffix(TRAINING_HISTORY_EXTENSION)


def get_models_evaluations_path(
    dir: pathlib.Path,
) -> pathlib.Path:
    return (dir / "models_evaluations").with_suffix(MODEL_EVALUATIONS_EXTENSION)


def save_generational_artifacts(
    generation_number: int,
    fittest: dict[cg.CompositeGenotype, gf.Fitness],
    rng: rand.RNG,
    novelty_tracker: gge.novelty.NoveltyTracker,
    output_dir: pathlib.Path,
) -> None:
    logger.info(
        f"started saving generation output, generation_number=<{generation_number}>"
    )

    if len(fittest) == 0:
        raise ValueError("`fittest` is empty")

    gen_out = GenerationOutput(
        generation_number=generation_number,
        fittest=fittest,
        novelty_tracker=novelty_tracker,
        rng=rng,
    )

    serialized_gen_out = pickle.dumps(gen_out, protocol=pickle.HIGHEST_PROTOCOL)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = get_generational_artifacts_path(generation_number, output_dir)
    path.write_bytes(serialized_gen_out)

    logger.info(
        f"finished saving generation output, generation_number=<{generation_number}>"
    )


def load_generational_artifacts(path: pathlib.Path) -> GenerationOutput:
    deserialized = pickle.loads(path.read_bytes())
    assert isinstance(deserialized, GenerationOutput)
    return deserialized


def load_latest_generational_artifacts(output_dir: pathlib.Path) -> GenerationOutput:
    paths = output_dir.glob(f"*{GENERATION_OUTPUT_EXTENSION}")
    latest = max(paths, key=lambda path: int(path.stem))
    return load_generational_artifacts(latest)


def save_genotype(genotype: cg.CompositeGenotype, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(genotype, protocol=pickle.HIGHEST_PROTOCOL))


def load_genotype(path: pathlib.Path) -> cg.CompositeGenotype:
    serialized = path.read_bytes()
    genotype = pickle.loads(serialized)
    assert isinstance(genotype, cg.CompositeGenotype)
    return genotype
