import uuid

import attrs

import gge.grammars.backbones as bb
import gge.grammars.structured_grammatical_evolution as sge
import gge.grammars.upper_grammars as ugr
import gge.neural_networks.connections as conn
import gge.randomness as rand


@attrs.frozen(cache_hash=True)
class CompositeGenotype:
    backbone_genotype: sge.Genotype = attrs.field(repr=False)
    connections_genotype: conn.ConnectionsSchema = attrs.field(repr=False)

    unique_id: uuid.UUID = attrs.field(
        eq=False, order=False, repr=True, init=False, factory=uuid.uuid4
    )

    def __attrs_post_init__(self) -> None:
        assert isinstance(self.backbone_genotype, sge.Genotype)
        assert isinstance(self.connections_genotype, conn.ConnectionsSchema)

    def __str__(self) -> str:
        return f"Genotype(UUID={self.unique_id.hex})"


def create_genotype(
    grammar: ugr.Grammar,
    rng: rand.RNG,
) -> CompositeGenotype:
    backbone_genotype = sge.create_genotype(grammar, rng)
    return make_composite_genotype(backbone_genotype, grammar, rng)


def make_composite_genotype(
    backbone_genotype: sge.Genotype,
    grammar: ugr.Grammar,
    rng: rand.RNG,
) -> CompositeGenotype:
    tokenstream = sge.map_to_tokenstream(backbone_genotype, grammar)
    backbone = bb.parse(tokenstream)
    connections_genotype = conn.create_connections_schema(backbone, rng)
    return CompositeGenotype(
        backbone_genotype,
        connections_genotype,
    )
