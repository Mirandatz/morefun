import networkx as nx
import tensorflow as tf

import gge.backbones as bb
import gge.composite_genotypes as cg
import gge.connections as conn
import gge.grammars as gr
import gge.layers as gl
import gge.optimizers as optim
import gge.structured_grammatical_evolution as sge


class NeuralNetwork:
    def __init__(
        self,
        output_layer: gl.ConnectableLayer,
        optimizer: optim.Optimizer,
    ) -> None:
        assert not isinstance(output_layer, gl.Input)

        graph = convert_to_digraph(output_layer)
        assert nx.is_directed_acyclic_graph(graph)

        layers = [node for node in graph.nodes]
        layer_names = [layer.name for layer in layers]
        assert len(set(layer_names)) == len(layer_names)

        inputs = [layer for layer in layers if isinstance(layer, gl.Input)]
        assert len(inputs) == 1

        self._input_layer = inputs[0]
        self._output_layer = output_layer
        self._optimizer = optimizer

    @property
    def input_layer(self) -> gl.Input:
        return self._input_layer

    @property
    def output_layer(self) -> gl.ConnectableLayer:
        return self._output_layer

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer

    def to_input_output_tensor(self) -> tuple[tf.Tensor, tf.Tensor]:
        tensores: dict[gl.ConnectableLayer, tf.Tensor] = {}
        self.output_layer.to_tensor(tensores)
        return tensores[self.input_layer], tensores[self.output_layer]


def make_network(
    genotype: cg.CompositeGenotype,
    grammar: gr.Grammar,
    input_layer: gl.Input,
) -> NeuralNetwork:
    tokenstream = sge.map_to_tokenstream(genotype.backbone_genotype, grammar)
    backbone = bb.parse(tokenstream)
    output_layer = conn.connect_backbone(
        backbone,
        genotype.connections_genotype,
        input_layer=input_layer,
    )
    optimizer = optim.parse(tokenstream)
    return NeuralNetwork(output_layer, optimizer)


def convert_to_digraph(output_layer: gl.ConnectableLayer) -> nx.DiGraph:
    graph = nx.DiGraph()
    to_visit = [output_layer]

    while to_visit:
        current = to_visit.pop()
        sources = reversed(list(gl.iter_sources(current)))
        for src in sources:
            graph.add_edge(src, current)
            to_visit.append(src)

    return graph
