import dataclasses

import typeguard

import gge.backbones as bb
import gge.connections as conn
import gge.randomness as rand
import gge.structured_grammatical_evolution as sge


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Genotype:
    backbone_genotype: sge.Genotype
    connections_genotype: conn.ConnectionsSchema


def create_genotype(backbone_genemancer: sge.Genemancer, rng: rand.RNG) -> Genotype:
    backbone_genotype = backbone_genemancer.create_genotype(rng)
    tokens = backbone_genemancer.map_to_tokens(backbone_genotype)
    tokenstream = "".join(tokens)
    backbone = bb.parse(tokenstream)
    connections_schema = conn.create_connections_schema(
        backbone=backbone,
        rng=rng,
    )
    return Genotype(backbone_genotype, connections_schema)
