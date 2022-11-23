import pathlib
import pickle

from loguru import logger

import gge.composite_genotypes as cg
import gge.evolutionary.fitnesses as gf
import gge.evolutionary.mutations as gm
import gge.evolutionary.novelty as novel
import gge.grammars.upper_grammars as ugr
import gge.paths
import gge.phenotypes as phenos
import gge.randomness as rand


class GenerationCheckpoint:
    """
    This class is used to allow us to stop running the evolutionary loop
    (e.g. by killing the Python interpreter itself) and resume its execution later.
    It should not be used to store results because it is sensitive to class/modules/packages
    refactors.
    """

    def __init__(
        self,
        generation_number: int,
        fitnesses: dict[cg.CompositeGenotype, gf.Fitness],
        novelty_tracker: novel.NoveltyTracker,
        rng: rand.RNG,
    ) -> None:
        self._generation_number = generation_number
        self._fitnesses = fitnesses.copy()
        self._novelty_tracker = novelty_tracker.copy()
        self._serialized_rng = pickle.dumps(rng, protocol=pickle.HIGHEST_PROTOCOL)

    def get_generation_number(self) -> int:
        return self._generation_number

    def get_fitnesses(self) -> dict[cg.CompositeGenotype, gf.Fitness]:
        return self._fitnesses.copy()

    def get_novelty_tracker(self) -> gge.evolutionary.novelty.NoveltyTracker:
        return self._novelty_tracker.copy()

    def get_rng(self) -> rand.RNG:
        rng = pickle.loads(self._serialized_rng)
        assert isinstance(rng, rand.RNG)
        return rng

    def save(self, path: pathlib.Path) -> None:
        """
        Serializes this instance and writes it to `path`.
        """
        serialized = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(serialized)

    @staticmethod
    def load(path: pathlib.Path) -> "GenerationCheckpoint":
        deserialized = pickle.loads(path.read_bytes())
        assert isinstance(deserialized, GenerationCheckpoint)
        return deserialized


def run_single_generation(
    population: dict[cg.CompositeGenotype, gf.Fitness],
    grammar: ugr.Grammar,
    mut_params: gm.PopulationMutationParameters,
    metrics: tuple[gf.Metric, ...],
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
) -> dict[cg.CompositeGenotype, gf.Fitness] | None:
    """
    Runs a single generation of the evolutionary loop and
    returns the fittest individuals genereated.

    If it is not possible to complete the generation process,
    (e.g., because the mutation procedure was unable to
    generate novelty individuals), this function returns
    `None` instead.
    """

    assert len(population) > 0

    initial_genotypes = list(population.keys())

    mutant_genotypes = gm.try_mutate_population(
        initial_genotypes,
        mut_params,
        rng,
        novelty_tracker,
    )

    if mutant_genotypes is None:
        return None

    evaluated_mutants = {
        genotype: gf.evaluate(phenos.translate(genotype, grammar), metrics)
        for genotype in mutant_genotypes
    }

    next_gen_candidates = population | evaluated_mutants
    fittest_genotypes: list[cg.CompositeGenotype] = gf.select_fittest_nsga2(
        next_gen_candidates, len(population)
    )

    return {g: next_gen_candidates[g] for g in fittest_genotypes}


def save_phenotypes(
    output_dir: pathlib.Path,
    fitnesses: dict[cg.CompositeGenotype, gf.Fitness],
) -> None:
    for genotype, fitness in fitnesses.items():
        if not fitness.successfully_evaluated():
            continue

        path = gge.paths.get_phenotype_path(output_dir, genotype.unique_id)
        path.write_bytes


def run_multiple_generations(
    starting_generation_number: int,
    number_of_generations_to_run: int,
    initial_population: dict[cg.CompositeGenotype, gf.Fitness],
    grammar: ugr.Grammar,
    mutation_params: gm.PopulationMutationParameters,
    metrics: tuple[gf.Metric, ...],
    novelty_tracker: novel.NoveltyTracker,
    rng: rand.RNG,
    output_dir: pathlib.Path,
) -> None:
    assert len(initial_population) > 0
    assert number_of_generations_to_run > 0

    output_dir.mkdir(parents=True, exist_ok=True)

    population = initial_population.copy()

    for gen_nr in range(
        starting_generation_number,
        starting_generation_number + number_of_generations_to_run,
    ):
        logger.info(f"started running generation {gen_nr}")

        maybe_population = run_single_generation(
            population=population,
            grammar=grammar,
            mut_params=mutation_params,
            metrics=metrics,
            rng=rng,
            novelty_tracker=novelty_tracker,
        )

        if maybe_population is None:
            logger.info(
                "stopping evolutionary loop: reason=<unable to generate enough novel mutants>"
            )
            break
        else:
            population = maybe_population

        GenerationCheckpoint(
            generation_number=gen_nr,
            fitnesses=population,
            rng=rng,
            novelty_tracker=novelty_tracker,
        ).save(gge.paths.get_generation_checkpoint_path(output_dir, gen_nr))

        logger.info(f"finished running generation {gen_nr}")
