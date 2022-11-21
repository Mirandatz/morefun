import pathlib
import pickle

from loguru import logger

import gge.composite_genotypes as cg
import gge.evolutionary.fitnesses as gf
import gge.evolutionary.novelty
import gge.paths
import gge.randomness as rand


class GenerationOutput:
    def __init__(
        self,
        generation_number: int,
        fittest: dict[cg.CompositeGenotype, gf.Fitness],
        novelty_tracker: gge.evolutionary.novelty.NoveltyTracker,
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

    def get_novelty_tracker(self) -> gge.evolutionary.novelty.NoveltyTracker:
        return self._novelty_tracker.copy()

    def get_rng(self) -> rand.RNG:
        rng = pickle.loads(self._serialized_rng)
        assert isinstance(rng, rand.RNG)
        return rng


def save_generational_artifacts(
    generation_number: int,
    fittest: dict[cg.CompositeGenotype, gf.Fitness],
    rng: rand.RNG,
    novelty_tracker: gge.evolutionary.novelty.NoveltyTracker,
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

    path = gge.paths.get_generation_output_path(output_dir, generation_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialized_gen_out)

    logger.info(
        f"finished saving generation output, generation_number=<{generation_number}>"
    )


def load_generational_artifacts(path: pathlib.Path) -> GenerationOutput:
    deserialized = pickle.loads(path.read_bytes())
    assert isinstance(deserialized, GenerationOutput)
    return deserialized


def load_latest_generational_artifacts(search_dir: pathlib.Path) -> GenerationOutput:
    path = gge.paths.get_latest_generation_output_path(search_dir)
    return load_generational_artifacts(path)
