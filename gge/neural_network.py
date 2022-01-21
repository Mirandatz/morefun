import networkx as nx
import tensorflow as tf

import gge.backbones as bb
import gge.composite_genotypes as cg
import gge.connections as conn
import gge.grammars as gr
import gge.layers as gl
import gge.name_generator as ng
import gge.structured_grammatical_evolution as sge


class NeuralNetwork:
    def __init__(self, output_layer: gl.ConnectableLayer) -> None:
        assert not isinstance(output_layer, gl.Input)

        graph = convert_to_digraph(output_layer)
        assert nx.is_directed_acyclic_graph(graph)

        inputs = [layer for layer in graph.nodes if isinstance(layer, gl.Input)]
        assert len(inputs) == 1

        self._input_layer = inputs[0]
        self._output_layer = output_layer

    @property
    def input_layer(self) -> gl.Input:
        return self._input_layer

    @property
    def output_layer(self) -> gl.ConnectableLayer:
        return self._output_layer

    def to_input_output_tensores(self) -> tuple[tf.Tensor, tf.Tensor]:
        tensores: dict[gl.ConnectableLayer, tf.Tensor] = {}
        self.output_layer.to_tensor(tensores)
        return tensores[self.input_layer], tensores[self.output_layer]


def make_network(
    genotype: cg.CompositeGenotype,
    grammar: gr.Grammar,
    input_layer: gl.Input,
) -> NeuralNetwork:
    backbone_tokenstream = sge.map_to_tokenstream(genotype.backbone_genotype, grammar)
    backbone = bb.parse(backbone_tokenstream)
    output_layer = conn.connect_backbone(
        backbone,
        genotype.connections_genotype,
        input_layer=input_layer,
    )
    return NeuralNetwork(output_layer)


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


def draw_graph(net: nx.DiGraph) -> None:
    assert isinstance(net, nx.DiGraph)
    import tempfile

    import cv2

    with tempfile.NamedTemporaryFile(prefix="/dev/shm/", suffix=".png") as file:
        agraph = nx.nx_agraph.to_agraph(net)
        agraph.draw(file.name, prog="dot")
        img = cv2.imread(file.name)
        cv2.imshow("network", img)
        cv2.waitKey()


def main() -> None:
    name_gen = ng.NameGenerator()
    input = gl.Input(gl.Shape(100, 100, 3))
    conv1 = gl.ConnectedConv2D(
        input,
        gl.Conv2D(name_gen.gen_name(gl.Conv2D), 128, 5, 2),
    )
    conv2 = gl.ConnectedConv2D(
        input,
        gl.Conv2D(name_gen.gen_name(gl.Conv2D), 128, 5, 4),
    )
    merge1 = conn.make_merge(
        [conv1, conv2],
        reshape_strategy=conn.ReshapeStrategy.DOWNSAMPLE,
        merge_strategy=conn.MergeStrategy.ADD,
        name_gen=name_gen,
    )
    graph = convert_to_digraph(merge1)
    draw_graph(graph)


if __name__ == "__main__":
    main()
