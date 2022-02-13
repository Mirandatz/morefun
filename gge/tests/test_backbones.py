import pytest
from hypothesis import example, given

import gge.backbones as bb
import gge.layers as gl
import gge.tests.strategies as gge_hs


@pytest.fixture(autouse=True)
def disable_logger() -> None:
    from loguru import logger

    logger.remove()


@given(gge_hs.backbone_grammar_layer(valid_layers=[gge_hs.conv2d_grammar_layer]))
@example(
    gge_hs.GrammarLayer(
        layers=(gl.Conv2D("Conv2D_0", 1, 2, 3),),
        mesagrammar_string='"conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3',
    )
)
@example(
    gge_hs.GrammarLayer(
        layers=(gl.make_merge("merge_0"), gl.Conv2D("Conv2D_0", 1, 2, 3)),
        mesagrammar_string='"merge" "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3',
    )
)
@example(
    gge_hs.GrammarLayer(
        layers=(gl.Conv2D("Conv2D_0", 1, 2, 3), gl.make_fork("fork_0")),
        mesagrammar_string='"conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "fork"',
    )
)
@example(
    gge_hs.GrammarLayer(
        layers=(
            gl.make_merge("merge_0"),
            gl.Conv2D("Conv2D_0", 1, 2, 3),
            gl.make_fork("fork_0"),
        ),
        mesagrammar_string='"merge" "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "fork"',
    )
)
@example(
    gge_hs.GrammarLayer(
        layers=(
            gl.Conv2D("Conv2D_0", 1, 2, 3),
            gl.Conv2D("Conv2D_1", 5, 6, 7),
            gl.Conv2D("Conv2D_2", 8, 9, 10),
        ),
        mesagrammar_string="""
    "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    "conv2d" "filter_count" 5 "kernel_size" 6 "stride" 7
    "conv2d" "filter_count" 8 "kernel_size" 9 "stride" 10
    """,
    )
)
@example(
    gge_hs.GrammarLayer(
        layers=(
            gl.Conv2D("Conv2D_0", 1, 2, 3),
            gl.make_fork("fork_0"),
            gl.make_merge("merge_0"),
            gl.Conv2D("Conv2D_1", 4, 5, 6),
            gl.make_fork("fork_1"),
            gl.make_merge("merge_1"),
            gl.Conv2D("Conv2D_2", 7, 8, 9),
            gl.Conv2D("Conv2D_3", 10, 11, 12),
            gl.Conv2D("Conv2D_4", 13, 14, 15),
            gl.make_merge("merge_2"),
            gl.Conv2D("Conv2D_5", 16, 17, 18),
        ),
        mesagrammar_string="""
    "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    "fork"

    "merge"
    "conv2d" "filter_count" 4 "kernel_size" 5 "stride" 6
    "fork"

    "merge"
    "conv2d" "filter_count" 7 "kernel_size" 8 "stride" 9

    "conv2d" "filter_count" 10 "kernel_size" 11 "stride" 12
    "conv2d" "filter_count" 13 "kernel_size" 14"stride" 15
    "merge"
    "conv2d" "filter_count" 16 "kernel_size" 17 "stride" 18
    """,
    )
)
def test_conv2d_backbone(backbone: gge_hs.GrammarLayer) -> None:
    """Can parse a backbone of Conv2D with merge and fork points."""
    actual = bb.parse(backbone.mesagrammar_string)
    expected = bb.Backbone(backbone.layers)
    assert actual == expected


@given(gge_hs.backbone_grammar_layer())
def test_parse_backbone(backbone: gge_hs.GrammarLayer) -> None:
    """Can parse a backbone with any layer, possibly marked."""
    actual = bb.parse(backbone.mesagrammar_string)
    expected = bb.Backbone(backbone.layers)
    assert actual == expected
