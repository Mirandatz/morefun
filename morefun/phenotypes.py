import pathlib
import pickle
import uuid

import attrs
import tensorflow as tf

import morefun.composite_genotypes as cg
import morefun.grammars.backbones as bb
import morefun.grammars.structured_grammatical_evolution as sge
import morefun.grammars.upper_grammars as ugr
import morefun.neural_networks.connections as conn
import morefun.neural_networks.layers as gl
import morefun.neural_networks.optimizers as optimizers
from morefun.grammars.backbones import parse as parse_backbone
from morefun.grammars.optimizers import parse as parse_optimizer


@attrs.frozen(cache_hash=True)
class Phenotype:
    """
    This class exists mostly to be tracked by the NoveltyTracker.
    """

    backbone: bb.Backbone = attrs.field(repr=False)
    connections: conn.ConnectionsSchema = attrs.field(repr=False)
    optimizer: optimizers.Optimizer = attrs.field(repr=False)

    genotype_uuid: uuid.UUID = attrs.field(eq=False, order=False, repr=True)

    def save(this, path: pathlib.Path) -> None:
        serialized = pickle.dumps(this, protocol=pickle.HIGHEST_PROTOCOL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(serialized)

    @staticmethod
    def load(path: pathlib.Path) -> "Phenotype":
        deserialized = pickle.loads(path.read_bytes())
        assert isinstance(deserialized, Phenotype)
        return deserialized


def translate(
    genotype: cg.CompositeGenotype,
    grammar: ugr.Grammar,
) -> Phenotype:
    tokenstream = sge.map_to_tokenstream(genotype.backbone_genotype, grammar)
    backbone = parse_backbone(tokenstream)
    optimizer = parse_optimizer(tokenstream)
    return Phenotype(
        backbone,
        genotype.connections_genotype,
        optimizer,
        genotype.unique_id,
    )


def make_input_output_tensors(
    phenotype: Phenotype,
    input_layer: gl.Input,
) -> tuple[tf.Tensor, tf.Tensor]:
    output_layer = conn.connect_backbone(
        phenotype.backbone,
        phenotype.connections,
        input_layer,
    )
    known_tensors: dict[gl.ConnectableLayer, tf.Tensor] = {}
    output_layer.to_tensor(known_tensors)
    return known_tensors[input_layer], known_tensors[output_layer]
