import functools
import typing

import attrs
from loguru import logger

import gge.composite_genotypes as cg
import gge.connections as conn
import gge.evolutionary.mutations as mutations
import gge.fallible as fallible
import gge.grammars.structured_grammatical_evolution as sge
import gge.grammars.upper_grammars as ugr
import gge.novelty as novel
import gge.phenotypes as gph
import gge.randomness as rand

T = typing.TypeVar("T")


def _update_tuple_item(
    tup: tuple[T, ...],
    index: int,
    new_value: T,
) -> tuple[T, ...]:
    assert len(tup) > 0
    assert 0 <= index < len(tup)

    temporary_container = list(tup)
    temporary_container[index] = new_value
    return tuple(temporary_container)


def _mutate_tuple(
    tup: tuple[T, ...],
    mutator: typing.Callable[[T], T],
    rng: rand.RNG,
) -> tuple[T, ...]:
    assert len(tup) > 0

    index = rng.integers(low=0, high=len(tup))
    old_value = tup[index]
    new_value = mutator(old_value)

    return _update_tuple_item(tup, index, new_value)


def mutate(
    genotype: cg.CompositeGenotype,
    grammar: ugr.Grammar,
    rng: rand.RNG,
) -> cg.CompositeGenotype:
    BACKBONE = "backbone"
    CONNECTIONS = "connections"
    options = [BACKBONE, CONNECTIONS]

    if genotype.connections_genotype.is_empty():
        logger.debug("ConnectionsGenotype is empty, removing mutation possibility.")
        options.remove(CONNECTIONS)

    chosen = rng.choice(options)
    logger.debug(f"Mutating individual on=<{chosen}>")

    if chosen == BACKBONE:
        skeleton = sge.make_genotype_skeleton(grammar)
        new_backbone_genotype = mutate_backbone_genotype(
            genotype.backbone_genotype, skeleton, rng
        )

        # maybe the evolutionary algorithm's locality could be improved
        # by trying to preserve as much of the original "connections genotype"
        # as possible, instead of generating an entirely new one
        return cg.make_composite_genotype(new_backbone_genotype, grammar, rng)

    else:
        new_connections = mutate_connections_schema(
            genotype.connections_genotype,
            rng,
        )
        return attrs.evolve(
            genotype,
            connections_genotype=new_connections,
        )


def mutate_backbone_genotype(
    backbone_genotype: sge.Genotype,
    skeleton: sge.GenotypeSkeleton,
    rng: rand.RNG,
) -> sge.Genotype:
    new_genes = _mutate_tuple(
        backbone_genotype.genes,
        mutator=functools.partial(mutate_gene, skeleton=skeleton, rng=rng),
        rng=rng,
    )
    return sge.Genotype(new_genes)


def mutate_gene(
    gene: sge.Gene,
    skeleton: sge.GenotypeSkeleton,
    rng: rand.RNG,
) -> sge.Gene:
    # maybe the evolutionary algorithm's locality could be improved by
    # trying to preserve as much of the original 'expansions indices' as possible,
    # instead of generating an entirely new one
    logger.trace("mutate_gene")
    return sge.create_gene(gene.nonterminal, skeleton, rng)


def mutate_connections_schema(
    schema: conn.ConnectionsSchema,
    rng: rand.RNG,
) -> conn.ConnectionsSchema:
    old_params = schema.merge_params

    new_params = _mutate_tuple(
        old_params,
        mutator=functools.partial(mutate_merge_parameters, rng=rng),
        rng=rng,
    )

    logger.debug(
        "Mutating connection schema from old=<{old_params}> to new=<{new_params}>"
    )

    return attrs.evolve(schema, merge_params=new_params)


def mutate_merge_parameters(
    params: conn.MergeParameters,
    rng: rand.RNG,
) -> conn.MergeParameters:
    options = ["forks", "reshape", "merge"]

    if len(params.forks_mask) == 0:
        options.remove("forks")

    mutation_type = rng.choice(options)

    if mutation_type == "forks":
        new_forks_mask = mutate_forks_mask(params.forks_mask, rng)
        return attrs.evolve(params, forks_mask=new_forks_mask)

    elif mutation_type == "reshape":
        new_reshape_strategy = mutate_reshape_strategy(params.reshape_strategy)
        return attrs.evolve(params, reshape_strategy=new_reshape_strategy)

    else:
        new_merge_strategy = mutate_merge_strategy(params.merge_strategy)
        return attrs.evolve(params, merge_strategy=new_merge_strategy)


def mutate_forks_mask(mask: tuple[bool, ...], rng: rand.RNG) -> tuple[bool, ...]:
    return _mutate_tuple(
        mask,
        mutator=lambda v: not v,
        rng=rng,
    )


def mutate_reshape_strategy(strat: conn.ReshapeStrategy) -> conn.ReshapeStrategy:
    if strat == conn.ReshapeStrategy.DOWNSAMPLE:
        return conn.ReshapeStrategy.UPSAMPLE

    elif strat == conn.ReshapeStrategy.UPSAMPLE:
        return conn.ReshapeStrategy.DOWNSAMPLE

    else:
        raise ValueError(f"unknown reshape strategy: {strat}")


def mutate_merge_strategy(strat: conn.MergeStrategy) -> conn.MergeStrategy:
    if strat == conn.MergeStrategy.ADD:
        return conn.MergeStrategy.CONCAT

    elif strat == conn.MergeStrategy.CONCAT:
        return conn.MergeStrategy.ADD

    else:
        raise ValueError(f"unknown merge strategy: {strat}")


@attrs.frozen
class PopulationMutationParameters:
    mutants_to_generate: int
    max_failures: int
    grammar: ugr.Grammar

    def __attrs_post_init__(self) -> None:
        assert self.mutants_to_generate >= 1
        assert self.max_failures >= 0


def try_generate_mutant(
    population: list[cg.CompositeGenotype],
    grammar: ugr.Grammar,
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
) -> cg.CompositeGenotype | None:
    logger.trace("try_generate_mutant")

    candidate: cg.CompositeGenotype = rng.choice(population)  # type: ignore

    mutant = mutations.mutate(candidate, grammar, rng)
    if novelty_tracker.is_genotype_novel(mutant):
        logger.debug(f"mutant has novel genotype=<{mutant}>")
        novelty_tracker.register_genotype(mutant)
    else:
        logger.debug(f"mutant has non-novel genotype=<{mutant}>")
        return None

    phenotype = gph.translate(mutant, grammar)
    if novelty_tracker.is_phenotype_novel(phenotype):
        logger.debug(f"mutant has novel phenotype=<{phenotype}>")
        novelty_tracker.register_phenotype(phenotype)
    else:
        logger.debug(f"mutant has non-novel phenotype=<{phenotype}>")
        return None

    return mutant


def try_mutate_population(
    population: list[cg.CompositeGenotype],
    mutation_params: PopulationMutationParameters,
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
) -> list[cg.CompositeGenotype] | None:
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
