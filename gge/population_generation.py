import functools

from loguru import logger

import gge.composite_genotypes as cg
import gge.fallible as fallible
import gge.grammars as gr
import gge.novelty as novel
import gge.phenotypes as gph
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
        logger.debug(f"created individual has novel genotype=<{genotype}>")
        novelty_tracker.register_genotype(genotype)
    else:
        logger.debug(f"created individual does not have novel genotype=<{genotype}>")
        return None

    phenotype = gph.translate(genotype, grammar)

    if novelty_tracker.is_phenotype_novel(phenotype):
        logger.debug(f"created individual has novel phenotype=<{phenotype}>")
        novelty_tracker.register_phenotype(phenotype)
    else:
        logger.debug(f"created individual does not have novel phenotype=<{phenotype}>")
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

    if initial_pop is None:
        msg = (
            f"unable to generate initial population with size=<{pop_size}>."
            f" reason: maximum failures=<{max_failures}> reached during novel-only individual creation"
        )
        raise ValueError(msg)

    return initial_pop
