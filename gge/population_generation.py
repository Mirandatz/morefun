import functools

from loguru import logger

import gge.composite_genotypes as cg
import gge.fallible as fallible
import gge.grammars as gr
import gge.layers as gl
import gge.neural_network as gnn
import gge.novelty as novel
import gge.randomness as rand


def try_create_individual(
    grammar: gr.Grammar,
    novelty_tracker: novel.NoveltyTracker,
    rng: rand.RNG,
) -> cg.CompositeGenotype | None:
    """
    Attempts to create a novel individual, that is,
    a genotype that is not yet tracked by `novelty_tracker`
    and whose phenotype is also not yet tracked by `novelty_tracker`.

    On success, the novel genotype is returned.
    If either the genotype or its phenotype is already tracked,
    `None` is returned instead.
    """

    logger.trace("try_create_individual")

    genotype = cg.create_genotype(grammar, rng)

    if novelty_tracker.is_genotype_novel(genotype):
        logger.debug(f"Created novel genotype for initial population=<{genotype}>")
        novelty_tracker.register_genotype(genotype)
    else:
        logger.debug("Failed to produce a novel individual")
        return None

    dummy_input = gl.make_input(1, 1, 1)
    phenotype = gnn.make_network(genotype, grammar, dummy_input)
    if novelty_tracker.is_phenotype_novel(phenotype):
        logger.debug(f"Created novel phenotype for initial population=<{phenotype}>")
        novelty_tracker.register_phenotype(phenotype)
    else:
        logger.debug("Failed to produce a novel phenotype")
        return None

    return genotype


def create_initial_population(
    grammar: gr.Grammar,
    pop_size: int,
    max_failures: int,
    novelty_tracker: novel.NoveltyTracker,
    rng: rand.RNG,
) -> list[cg.CompositeGenotype]:
    assert pop_size > 0
    assert max_failures >= 0

    generator = functools.partial(
        try_create_individual,
        grammar=grammar,
        novelty_tracker=novelty_tracker,
        rng=rng,
    )

    initial_pop = fallible.collect_results_from_fallible_function(
        generator,
        num_results=pop_size,
        max_failures=max_failures,
    )

    assert initial_pop is not None
    return initial_pop
