import functools

import gge.composite_genotypes as cg
import gge.fallible as fallible
import gge.mutations as mutations
import gge.novelty as novel
import gge.randomness as rand
import gge.structured_grammatical_evolution as sge


def try_generate_mutant(
    pop: list[cg.CompositeGenotype],
    genemancer: sge.Genemancer,
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
) -> cg.CompositeGenotype | None:
    candidate = rng.choice(pop)  # type: ignore
    mutant = mutations.mutate(candidate, genemancer, rng)
    if not novelty_tracker.is_genotype_novel(mutant):
        return None

    novelty_tracker.register_genotype(mutant)
    return mutant


def try_mutate_population(
    pop: list[cg.CompositeGenotype],
    mutants_to_generate: int,
    max_failures: int,
    genemancer: sge.Genemancer,
    rng: rand.RNG,
    novelty_tracker: novel.NoveltyTracker,
) -> list[cg.CompositeGenotype] | None:
    assert len(pop) > 0
    assert mutants_to_generate > 1
    assert max_failures >= 0

    # we only update the actual tracker if we succeed
    tracker_copy = novelty_tracker.copy()

    generator = functools.partial(
        try_generate_mutant,
        pop,
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
