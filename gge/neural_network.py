import attrs
import networkx as nx
import tensorflow as tf

import gge.backbones as bb
import gge.composite_genotypes as cg
import gge.connections as conn
import gge.grammars as gr
import gge.layers as gl
import gge.optimizers as optim
import gge.structured_grammatical_evolution as sge


@attrs.frozen(kw_only=True)
class NeuralNetwork:
    input_layer: gl.Input
    output_layer: gl.ConnectableLayer
    optimizer: optim.Optimizer

    def __attrs_post_init__(self) -> None:
        validate_layers(self.input_layer, self.output_layer)

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

    return NeuralNetwork(
        input_layer=input_layer,
        output_layer=output_layer,
        optimizer=optimizer,
    )


def validate_layers(
    input_layer: gl.Input,
    output_layer: gl.ConnectableLayer,
) -> None:
    graph = convert_to_digraph(output_layer)
    if not nx.is_directed_acyclic_graph(graph):
        # TODO: pickle output_layer to enable postmortem debug
        raise ValueError(
            "the graph described by tracing `output_layer`'s inputs is cyclic"
        )

    layers = [node for node in graph.nodes]

    validate_input_layers(input_layer, layers)
    validate_layer_names(layers)


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


def validate_layer_names(layers: list[gl.ConnectableLayer]) -> None:
    layer_names = [layer.name for layer in layers]
    unique_names = set()
    repeated_names = set()

    for name in layer_names:
        if name not in unique_names:
            unique_names.add(name)
        else:
            repeated_names.add(name)

    if repeated_names:
        raise ValueError(
            f"a network must contain only uniquely named layers, repeated names=<{repeated_names}>"
        )


def validate_input_layers(
    input_layer: gl.Input,
    layers: list[gl.ConnectableLayer],
) -> None:
    inputs = [layer for layer in layers if isinstance(layer, gl.Input)]
    if len(inputs) != 1:
        raise ValueError("the network must contain one and only one`Input` layer")

    if inputs[0] != input_layer:
        raise ValueError(
            "mismatch between provided `input_layer` and the one found by"
            " tracing the inputs back"
        )
