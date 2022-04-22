import uuid

import attrs

import gge.backbones as bb
import gge.connections as conn
import gge.grammars as gr
import gge.randomness as rand
import gge.structured_grammatical_evolution as sge


@attrs.frozen(cache_hash=True)
class CompositeGenotype:
    backbone_genotype: sge.Genotype = attrs.field(repr=False)
    connections_genotype: conn.ConnectionsSchema = attrs.field(repr=False)

    unique_id: uuid.UUID = attrs.field(
        eq=False, order=False, repr=True, init=False, default=uuid.uuid4()
    )

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.backbone_genotype, sge.Genotype)
        assert isinstance(self.connections_genotype, conn.ConnectionsSchema)


def create_genotype(
    grammar: gr.Grammar,
    rng: rand.RNG,
) -> CompositeGenotype:
    backbone_genotype = sge.create_genotype(grammar, rng)
    return make_composite_genotype(backbone_genotype, grammar, rng)


def make_composite_genotype(
    backbone_genotype: sge.Genotype,
    grammar: gr.Grammar,
    rng: rand.RNG,
) -> CompositeGenotype:
    tokenstream = sge.map_to_tokenstream(backbone_genotype, grammar)
    backbone = bb.parse(tokenstream, start="start")
    connections_genotype = conn.create_connections_schema(backbone, rng)
    return CompositeGenotype(
        backbone_genotype,
        connections_genotype,
    )
