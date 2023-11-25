import morefun.composite_genotypes as cg
import morefun.phenotypes as ph


class NoveltyTracker:
    def __init__(
        self,
        known_genotypes: set[cg.CompositeGenotype] | None = None,
        known_phenotypes: set[ph.Phenotype] | None = None,
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

    def is_phenotype_novel(self, phenotype: ph.Phenotype) -> bool:
        return phenotype not in self._phenotypes

    def register_genotype(self, genotype: cg.CompositeGenotype) -> None:
        self._genotypes.add(genotype)

    def register_phenotype(self, phenotype: ph.Phenotype) -> None:
        self._phenotypes.add(phenotype)
