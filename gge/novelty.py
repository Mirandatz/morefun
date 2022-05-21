import gge.composite_genotypes as cg
import gge.phenotypes as phenos


class NoveltyTracker:
    def __init__(
        self,
        known_genotypes: set[cg.CompositeGenotype] | None = None,
        known_phenotypes: set[phenos.Phenotype] | None = None,
    ) -> None:
        self._genotypes = (known_genotypes or set()).copy()
        self._phenotypes = (known_phenotypes or set()).copy()

    def copy(self) -> "NoveltyTracker":
        return NoveltyTracker(
            known_genotypes=self._genotypes,
            known_phenotypes=self._phenotypes,
        )

    def update(self, other: "NoveltyTracker") -> None:
        self._genotypes.update(other._genotypes)
        self._phenotypes.update(other._phenotypes)

    def is_genotype_novel(self, genotype: cg.CompositeGenotype) -> bool:
        return genotype not in self._genotypes

    def is_phenotype_novel(self, phenotype: phenos.Phenotype) -> bool:
        return phenotype not in self._phenotypes

    def register_genotype(self, genotype: cg.CompositeGenotype) -> None:
        self._genotypes.add(genotype)

    def register_phenotype(self, phenotype: phenos.Phenotype) -> None:
        self._phenotypes.add(phenotype)


def check_novelty_and_update_tracker(
    genotype: cg.CompositeGenotype,
    phenotype: phenos.Phenotype,
    tracker: NoveltyTracker,
) -> bool:
    if not tracker.is_genotype_novel(genotype):
        return False

    if not tracker.is_phenotype_novel(phenotype):
        # got a new genotype that maps to a known phenotype,
        # so we mark the genotype as known and return false
        tracker.register_genotype(genotype)
        return False

    else:
        # found a new genotype that maps to new phenotype,
        # so we mark both as known and return true
        tracker.register_genotype(genotype)
        tracker.register_phenotype(phenotype)
        return True
