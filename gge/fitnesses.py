import dataclasses
import enum
import typing

import gge.composite_genotypes as cg
import gge.grammars as gr
import gge.layers as gl
import gge.neural_network as gnn


@enum.unique
class FitnessMetric(enum.Enum):
    layer_count = enum.auto()


@dataclasses.dataclass(frozen=True)
class FitnessEvaluationParameters:
    metric: FitnessMetric
    grammar: gr.Grammar
    input_layer: gl.Input


def layer_count(model: gnn.NeuralNetwork) -> float:
    graph = gnn.convert_to_digraph(model)
    layers = graph.nodes
    cool_layers = [
        layer
        for layer in layers
        if isinstance(layer, gl.SingleInputLayer | gl.MultiInputLayer)
    ]
    return len(cool_layers)


def evaluate(
    genotype: cg.CompositeGenotype,
    params: FitnessEvaluationParameters,
) -> float:

    phenotype = gnn.make_network(
        genotype,
        params.grammar,
        params.input_layer,
    )

    if params.metric == FitnessMetric.layer_count:
        return layer_count(phenotype)

    else:
        raise ValueError(f"unknown evaluation metric: {params.metric}")


T = typing.TypeVar("T")


def select_fittest(
    population: dict[T, float],
    fittest_count: int,
) -> dict[T, float]:
    """
    This function assumes that fitness must be maximized.
    """
    assert len(population) >= fittest_count
    assert fittest_count > 0

    best_to_worst = sorted(
        population.keys(),
        key=lambda g: population[g],
        reverse=True,
    )

    fittest = best_to_worst[:fittest_count]
    return {g: population[g] for g in fittest}
