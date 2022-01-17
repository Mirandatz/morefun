import gge.composite_genotypes as cg
import gge.neural_network as gnn


class NoveltyTracker:
    def __init__(
        self,
        known_genotypes: set[cg.CompositeGenotype] = set(),
        known_phenotypes: set[gnn.NeuralNetwork] = set(),
    ) -> None:
        self._genotypes = known_genotypes.copy()
        self._phenotypes = known_phenotypes.copy()

    def copy(self) -> "NoveltyTracker":
        return NoveltyTracker(
            known_genotypes=self._genotypes,
            known_phenotypes=self._phenotypes,
        )

    def update(self, other: "NoveltyTracker") -> None:
        self._genotypes.update(other._genotypes)
        self._phenotypes.update(other._phenotypes)

    def is_genotype_novel(self, genotype: cg.CompositeGenotype) -> bool:
        return genotype in self._genotypes

    def is_phenotype_novel(self, phenotype: gnn.NeuralNetwork) -> bool:
        return phenotype in self._phenotypes

    def register_genotype(self, genotype: cg.CompositeGenotype) -> None:
        self._genotypes.add(genotype)

    def register_phenotype(self, phenotype: gnn.NeuralNetwork) -> None:
        self._phenotypes.add(phenotype)
