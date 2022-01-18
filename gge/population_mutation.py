import functools

import gge.composite_genotypes as cg
import gge.fallible as fallible
import gge.layers as gl
import gge.mutations as mutations
import gge.neural_network as gnn
import gge.novelty as novel
import gge.randomness as rand
import gge.structured_grammatical_evolution as sge

DUMMY_INPUT = gl.make_input(1, 1, 1)


def try_generate_mutant(
    population: list[cg.CompositeGenotype],
    genemancer: sge.Genemancer,
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
) -> cg.CompositeGenotype | None:

    candidate: cg.CompositeGenotype = rng.choice(population)  # type: ignore

    mutant = mutations.mutate(candidate, genemancer, rng)
    if novelty_tracker.is_genotype_novel(mutant):
        novelty_tracker.register_genotype(mutant)
    else:
        return None

    phenotype = gnn.make_network(mutant, genemancer, DUMMY_INPUT)
    if novelty_tracker.is_phenotype_novel(phenotype):
        novelty_tracker.register_phenotype(phenotype)
    else:
        return None

    return mutant


def try_mutate_population(
    population: list[cg.CompositeGenotype],
    mutants_to_generate: int,
    max_failures: int,
    genemancer: sge.Genemancer,
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
) -> list[cg.CompositeGenotype] | None:
    assert len(population) > 0
    assert mutants_to_generate > 1
    assert max_failures >= 0

    # we only update the actual tracker if we succeed
    tracker_copy = novelty_tracker.copy()

    generator = functools.partial(
        try_generate_mutant,
        population,
        genemancer,
        rng,
        tracker_copy,
    )

    results = fallible.collect_results_from_fallible_function(
        generator,
        num_results=mutants_to_generate,
        max_failures=max_failures,
    )

    if results is None:
        return None

    novelty_tracker.update(tracker_copy)
    return results
