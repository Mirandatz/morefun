import dataclasses

import gge.backbones as bb
import gge.connections as conn
import gge.grammars as gr
import gge.randomness as rand
import gge.structured_grammatical_evolution as sge


@dataclasses.dataclass(frozen=True)
class CompositeGenotype:
    backbone_genotype: sge.Genotype
    connections_genotype: conn.ConnectionsSchema

    def __post_init__(self) -> None:
        assert isinstance(self.backbone_genotype, sge.Genotype)
        assert isinstance(self.connections_genotype, conn.ConnectionsSchema)


def create_genotype(
    grammar: gr.Grammar,
    rng: rand.RNG,
) -> CompositeGenotype:
    backbone_genotype = sge.create_genotype(grammar, rng)
    tokenstream = sge.map_to_tokenstream(backbone_genotype, grammar)
    backbone = bb.parse(tokenstream)
    connections_schema = conn.create_connections_schema(
        backbone=backbone,
        rng=rng,
    )
    return CompositeGenotype(backbone_genotype, connections_schema)


def make_composite_genotype(
    backbone_genotype: sge.Genotype,
    grammar: gr.Grammar,
    rng: rand.RNG,
) -> CompositeGenotype:
    tokenstream = sge.map_to_tokenstream(backbone_genotype, grammar)
    backbone = bb.parse(tokenstream)
    connections_genotype = conn.create_connections_schema(backbone, rng)
    return CompositeGenotype(
        backbone_genotype,
        connections_genotype,
    )
