import gge.backbones as bb
import gge.grammars as gr
import gge.layers as gl
import gge.randomness as rand
import gge.structured_grammatical_evolution as sge


def make_backbone(raw_grammar: str) -> bb.Backbone:
    grammar = gr.Grammar(raw_grammar)
    genemancer = sge.Genemancer(grammar)
    rng = rand.create_rng()
    genotype = genemancer.create_genotype(rng)
    tokenstream = genemancer.map_to_tokenstream(genotype)
    return bb.parse(tokenstream)


def test_conv() -> None:
    backbone = make_backbone(
        """
        start : "conv2d" "filter_count" 64 "kernel_size" 5 "stride" 2
        """
    )

    (conv,) = backbone.layers

    assert isinstance(conv, gl.Conv2D)
    assert 64 == conv.filter_count
    assert 5 == conv.kernel_size
    assert 2 == conv.stride


def test_relu() -> None:
    backbone = make_backbone(
        """
        start : "relu"
        """
    )

    (relu,) = backbone.layers
    assert isinstance(relu, gl.Relu)


def test_conv_relu() -> None:
    backbone = make_backbone(
        """
        start : feature act
        feature: "conv2d" "filter_count" 64 "kernel_size" 5 "stride" 2
        act: "relu"
        """
    )

    conv, relu = backbone.layers

    assert isinstance(conv, gl.Conv2D)
    assert 64 == conv.filter_count
    assert 5 == conv.kernel_size
    assert 2 == conv.stride

    assert isinstance(relu, gl.Relu)


def test_merge_relu() -> None:
    backbone = make_backbone(
        """
        start : "merge" act
        act   : "relu"
        """
    )

    merge, act = backbone.layers
    assert gl.is_merge_marker(merge)
    assert isinstance(act, gl.Relu)
