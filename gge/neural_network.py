import networkx as nx

import gge.connections as conn
import gge.layers as gl
import gge.name_generator as ng


class NeuralNetwork:
    ...


def make_network(output_layer: gl.ConnectableLayer) -> nx.DiGraph:
    assert not isinstance(output_layer, gl.Input)

    network = nx.DiGraph()
    to_visit = [output_layer]

    while to_visit:
        current = to_visit.pop()
        sources = reversed(list(gl.iter_sources(current)))
        for src in sources:
            network.add_edge(src, current)
            to_visit.append(src)

    return network


def draw_network(net: nx.DiGraph) -> None:
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
    network = make_network(merge1)
    draw_network(network)


if __name__ == "__main__":
    main()
