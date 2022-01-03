import networkx as nx

import gge.layers as gl


def make_network(output_layer: gl.ConnectedLayer) -> nx.DiGraph:
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
    input = gl.Input(gl.Shape(10, 10, 3))
    conv1 = gl.ConnectedConv2D(input, gl.Conv2D("c1", 128, 5, 2))
    conv2 = gl.ConnectedConv2D(input, gl.Conv2D("c2", 128, 5, 2))
    add1 = gl.ConnectedAdd((conv1, conv2))
    network = make_network(add1)
    draw_network(network)


if __name__ == "__main__":
    main()
