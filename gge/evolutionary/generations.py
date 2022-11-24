import pathlib
import pickle

import attrs
from loguru import logger

import gge.composite_genotypes as cg
import gge.evolutionary.fitnesses as gf
import gge.evolutionary.mutations as gm
import gge.evolutionary.novelty as novel
import gge.grammars.upper_grammars as ugr
import gge.paths
import gge.phenotypes as phenos
import gge.randomness as rand


@attrs.frozen
class EvaluatedGenotype:
    genotype: cg.CompositeGenotype
    phenotype: phenos.Phenotype
    fitness: gf.Fitness


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
        population: tuple[EvaluatedGenotype, ...],
        novelty_tracker: novel.NoveltyTracker,
        rng: rand.RNG,
    ) -> None:
        assert generation_number >= 0
        assert len(population) >= 1

        self._generation_number = generation_number
        self._population = population
        self._novelty_tracker = novelty_tracker.copy()
        self._serialized_rng = pickle.dumps(rng, protocol=pickle.HIGHEST_PROTOCOL)

    def get_generation_number(self) -> int:
        return self._generation_number

    def get_population(self) -> tuple[EvaluatedGenotype, ...]:
        return self._population

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
    generation_number: int,
    population: tuple[EvaluatedGenotype, ...],
    grammar: ugr.Grammar,
    mutation_params: gm.PopulationMutationParameters,
    fitness_metrics: tuple[gf.Metric, ...],
    novelty_tracker: novel.NoveltyTracker,
    rng: rand.RNG,
) -> GenerationCheckpoint | None:
    """
    Runs a single generation of the evolutionary loop.

    If it is not possible to complete the generation process
    (e.g., because the mutation procedure was unable to  generate novelty individuals)
    this function returns `None`.
    """
    assert generation_number >= 0
    assert len(population) >= 1

    logger.info(f"started running generation={generation_number}")

    initial_genotypes = [x.genotype for x in population]

    mutant_genotypes = gm.try_mutate_population(
        initial_genotypes,
        mutation_params,
        rng,
        novelty_tracker,
    )

    if mutant_genotypes is None:
        return None

    mutant_phenotypes = [phenos.translate(g, grammar) for g in mutant_genotypes]
    mutant_fitnesses = [gf.evaluate(p, fitness_metrics) for p in mutant_phenotypes]
    evaluated_mutants = [
        EvaluatedGenotype(g, p, f)
        for g, p, f in zip(
            mutant_genotypes,
            mutant_phenotypes,
            mutant_fitnesses,
        )
    ]

    next_gen_candidates = {e: e.fitness for e in evaluated_mutants} | {
        e: e.fitness for e in population
    }

    fittest_genotypes = gf.select_fittest_nsga2(
        evaluated_solutions=next_gen_candidates,
        fittest_count=len(population),
    )

    checkpoint = GenerationCheckpoint(
        generation_number=generation_number,
        population=tuple(fittest_genotypes),
        novelty_tracker=novelty_tracker,
        rng=rng,
    )

    logger.info(f"finished running generation=<{generation_number}>")

    return checkpoint


def run_multiple_generations(
    starting_generation_number: int,
    number_of_generations_to_run: int,
    initial_population: tuple[EvaluatedGenotype, ...],
    grammar: ugr.Grammar,
    mutation_params: gm.PopulationMutationParameters,
    metrics: tuple[gf.Metric, ...],
    novelty_tracker: novel.NoveltyTracker,
    rng: rand.RNG,
    output_dir: pathlib.Path,
) -> None:
    assert len(initial_population) >= 1
    assert number_of_generations_to_run >= 1

    checkpoint = GenerationCheckpoint(
        generation_number=starting_generation_number,
        population=initial_population,
        novelty_tracker=novelty_tracker,
        rng=rng,
    )

    for _ in range(number_of_generations_to_run):
        maybe_checkpoint = run_single_generation(
            generation_number=checkpoint.get_generation_number() + 1,
            population=checkpoint.get_population(),
            grammar=grammar,
            mutation_params=mutation_params,
            fitness_metrics=metrics,
            rng=checkpoint.get_rng(),
            novelty_tracker=checkpoint.get_novelty_tracker(),
        )

        if maybe_checkpoint is None:
            logger.info(
                "stopping evolutionary loop: reason=<unable to generate enough novel mutants>"
            )
            break

        else:
            checkpoint = maybe_checkpoint

        save_path = gge.paths.get_generation_checkpoint_path(
            output_dir,
            checkpoint.get_generation_number(),
        )

        checkpoint.save(save_path)
