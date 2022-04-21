import uuid

import attrs

import gge.backbones as bb
import gge.composite_genotypes as cg
import gge.connections as conn
import gge.data_augmentations as gda
import gge.grammars as gr
import gge.optimizers as optim
import gge.structured_grammatical_evolution as sge


@attrs.frozen(cache_hash=True)
class Phenotype:
    """
    This class exists mostly to be tracked by the NoveltyTracker.
    """

    data_augmentation: gda.DataAugmentation = attrs.field(repr=False)
    backbone: bb.Backbone = attrs.field(repr=False)
    connections: conn.ConnectionsSchema = attrs.field(repr=False)
    optimizer: optim.Optimizer = attrs.field(repr=False)

    genotype_uuid: uuid.UUID = attrs.field(eq=False, order=False, repr=True)


def translate(
    genotype: cg.CompositeGenotype,
    grammar: gr.Grammar,
) -> Phenotype:
    tokenstream = sge.map_to_tokenstream(genotype.backbone_genotype, grammar)
    data_aug = gda.parse(tokenstream, start="start")
    backbone = bb.parse(tokenstream, start="start")
    optimizer = optim.parse(tokenstream, start="start")
    return Phenotype(
        data_aug,
        backbone,
        genotype.connections_genotype,
        optimizer,
        genotype.unique_id,
    )
