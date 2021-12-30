import dataclasses

import networkx as nx
import typeguard

import gge.backbones as bb
import gge.connections as conn


def make_network(
    backbone: bb.Backbone,
    connections: conn.ConnectionsSchema,
) -> nx.DiGraph:
    raise NotImplementedError()


def main() -> None:
    pass


if __name__ == "__main__":
    main()
