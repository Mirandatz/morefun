import dataclasses

import networkx as nx

import gge.backbones as bb
import gge.composite_genotypes as cg
import gge.connections as conn
import gge.grammars as gr
import gge.layers as gl
import gge.name_generator as ng
import gge.structured_grammatical_evolution as sge


@dataclasses.dataclass(frozen=True)
class NeuralNetwork:
    output_layer: gl.ConnectableLayer

    def __post_init__(self) -> None:
        assert not isinstance(self.output_layer, gl.Input)

        graph = convert_to_digraph(self)

        assert nx.is_directed_acyclic_graph(graph)

        inputs = [layer for layer in graph.nodes if isinstance(layer, gl.Input)]
        assert len(inputs) == 1


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


def convert_to_digraph(network: NeuralNetwork) -> nx.DiGraph:
    graph = nx.DiGraph()
    to_visit = [network.output_layer]

    while to_visit:
        current = to_visit.pop()
        sources = reversed(list(gl.iter_sources(current)))
        for src in sources:
            graph.add_edge(src, current)
            to_visit.append(src)

    return graph


def draw_graph(net: nx.DiGraph) -> None:
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
    graph = convert_to_digraph(NeuralNetwork(merge1))
    draw_graph(graph)


if __name__ == "__main__":
    main()
