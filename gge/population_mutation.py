import dataclasses
import functools
import typing

from loguru import logger

import gge.composite_genotypes as cg
import gge.fallible as fallible
import gge.grammars as gr
import gge.layers as gl
import gge.mutations as mutations
import gge.neural_network as gnn
import gge.novelty as novel
import gge.randomness as rand

DUMMY_INPUT = gl.make_input(1, 1, 1)


@dataclasses.dataclass(frozen=True)
class PopulationMutationParameters:
    mutants_to_generate: int
    max_failures: int
    grammar: gr.Grammar

    def __post_init__(self) -> None:
        assert self.mutants_to_generate > 1
        assert self.max_failures >= 0


def try_generate_mutant(
    population: list[cg.CompositeGenotype],
    grammar: gr.Grammar,
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
) -> typing.Optional[cg.CompositeGenotype]:
    logger.trace("try_generate_mutant")

    candidate: cg.CompositeGenotype = rng.choice(population)  # type: ignore

    mutant = mutations.mutate(candidate, grammar, rng)
    if novelty_tracker.is_genotype_novel(mutant):
        logger.debug(f"New novelty mutant=<{mutant}>")
        novelty_tracker.register_genotype(mutant)
    else:
        logger.debug("Failed to produce a novelty individual")
        return None

    phenotype = gnn.make_network(mutant, grammar, DUMMY_INPUT)
    if novelty_tracker.is_phenotype_novel(phenotype):
        logger.debug(f"New novelty phenotype=<{phenotype}>")
        novelty_tracker.register_phenotype(phenotype)
    else:
        logger.debug("Failed to produce a novelty phenotype")
        return None

    return mutant


def try_mutate_population(
    population: list[cg.CompositeGenotype],
    mutation_params: PopulationMutationParameters,
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
) -> typing.Optional[list[cg.CompositeGenotype]]:
    assert len(population) > 0

    # we only update the actual tracker if we succeed
    tracker_copy = novelty_tracker.copy()

    generator = functools.partial(
        try_generate_mutant,
        population,
        mutation_params.grammar,
        rng,
        tracker_copy,
    )

    results = fallible.collect_results_from_fallible_function(
        generator,
        num_results=mutation_params.mutants_to_generate,
        max_failures=mutation_params.max_failures,
    )

    if results is None:
        return None

    novelty_tracker.update(tracker_copy)
    return results
