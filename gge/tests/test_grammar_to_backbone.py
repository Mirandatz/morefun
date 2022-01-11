import hypothesis
from hypothesis import given
import hypothesis.strategies as hs

import gge.backbones as bb
import gge.grammars as gr
import gge.layers as gl
import gge.randomness as rand
import gge.structured_grammatical_evolution as sge
import gge.tests.strategies as gge_hs


# end-to-end tests may take a while
end_to_end_settings = hypothesis.settings(max_examples=50)


def make_backbone(raw_grammar: str, rng_seed: int | None = None) -> bb.Backbone:
    grammar = gr.Grammar(raw_grammar)
    genemancer = sge.Genemancer(grammar)
    rng = rand.create_rng(rng_seed)
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


@given(seed=hs.integers(min_value=1), filter_counts=gge_hs.grammar_integer_option())
@end_to_end_settings
def test_multiple_option(seed: int, filter_counts: gge_hs.GrammarOption) -> None:
    """Grammar accepts multiple integer arguments for constants"""
    backbone = make_backbone(
        f"""
        start : "conv2d" "filter_count" {filter_counts.mesagrammar_string} "kernel_size" 5 "stride" 2
        """,
        seed,
    )

    (conv,) = backbone.layers

    assert conv.filter_count in filter_counts.possible_values


@given(
    seed=hs.integers(min_value=1),
    filter_counts=gge_hs.grammar_integer_option(),
    strides=gge_hs.grammar_integer_option(),
)
@end_to_end_settings
def test_multiple_option_many_places(
    seed: int, filter_counts: gge_hs.GrammarOption, strides: gge_hs.GrammarOption
) -> None:
    """Grammar accepts multiple integer arguments for constants at different places of the grammar"""
    backbone = make_backbone(
        f"""
        start : "conv2d" "filter_count" {filter_counts.mesagrammar_string} "kernel_size" 5 "stride" {strides.mesagrammar_string}
        """,
        seed,
    )

    (conv,) = backbone.layers

    assert conv.filter_count in filter_counts.possible_values
    assert conv.stride in strides.possible_values
