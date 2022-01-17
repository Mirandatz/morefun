import gge.composite_genotypes as cg
import gge.neural_network as gnn


class NoveltyTracker:
    def __init__(self) -> None:
        self._known_genotypes: set[cg.CompositeGenotype] = set()
        self._known_phenotypes: set[gnn.NeuralNetwork] = set()

    def is_genotype_novel(self, genotype: cg.CompositeGenotype) -> bool:
        return genotype in self._known_genotypes

    def is_phenotype_novel(self, phenotype: gnn.NeuralNetwork) -> bool:
        return phenotype in self._known_phenotypes

    def register_known_genotype(self, genotype: cg.CompositeGenotype) -> None:
        if genotype in self._known_genotypes:
            raise ValueError("genotype is already known")

        self._known_genotypes.add(genotype)

    def register_phenotype(self, phenotype: gnn.NeuralNetwork) -> None:
        if phenotype in self._known_phenotypes:
            raise ValueError("phenotype is already known")

        self._known_phenotypes.add(phenotype)
