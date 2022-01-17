import gge.composite_genotypes as cg
import gge.neural_network as gnn


class NoveltyTracker:
    def __init__(self) -> None:
        raise NotImplementedError()

    def is_genotype_novel(self, genotype: cg.CompositeGenotype) -> bool:
        raise NotImplementedError()

    def is_phenotype_novel(self, phenotype: gnn.NeuralNetwork) -> bool:
        raise NotImplementedError()

    def register_known_genotype(self, genotype: cg.CompositeGenotype) -> None:
        raise NotImplementedError()

    def register_phenotype(self, phenotype: gnn.NeuralNetwork) -> None:
        raise NotImplementedError()
