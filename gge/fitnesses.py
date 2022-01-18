import typing

import gge.layers as gl
import gge.neural_network as gnn


def dummy_fitness_evaluation(model: gnn.NeuralNetwork) -> float:
    graph = gnn.convert_to_digraph(model)
    layers = graph.nodes
    cool_layers = [layer for layer in layers if isinstance(layer, gl.SingleInputLayer)]
    return len(cool_layers)


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
