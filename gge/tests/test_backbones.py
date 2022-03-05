import pytest
from hypothesis import example, given  # noqa

import gge.backbones as bb
import gge.tests.strategies.mesagrammar as mesa


@pytest.fixture(autouse=True)
def disable_logger() -> None:
    from loguru import logger

    logger.remove()


# @given(mesa.backbone_grammar_layer(valid_layers=[mesa.conv2d_grammar_layer]))
# @example(
#     mesa.LayersTestData(
#         layers=(gl.Conv2D("Conv2D_0", 1, 2, 3),),
#         mesagrammar_string='"conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3',
#     )
# )
# @example(
#     mesa.LayersTestData(
#         layers=(gl.make_merge("merge_0"), gl.Conv2D("Conv2D_0", 1, 2, 3)),
#         mesagrammar_string='"merge" "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3',
#     )
# )
# @example(
#     mesa.LayersTestData(
#         layers=(gl.Conv2D("Conv2D_0", 1, 2, 3), gl.make_fork("fork_0")),
#         mesagrammar_string='"conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "fork"',
#     )
# )
# @example(
#     mesa.LayersTestData(
#         layers=(
#             gl.make_merge("merge_0"),
#             gl.Conv2D("Conv2D_0", 1, 2, 3),
#             gl.make_fork("fork_0"),
#         ),
#         mesagrammar_string='"merge" "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "fork"',
#     )
# )
# @example(
#     mesa.LayersTestData(
#         layers=(
#             gl.Conv2D("Conv2D_0", 1, 2, 3),
#             gl.Conv2D("Conv2D_1", 5, 6, 7),
#             gl.Conv2D("Conv2D_2", 8, 9, 10),
#         ),
#         mesagrammar_string="""
#     "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
#     "conv2d" "filter_count" 5 "kernel_size" 6 "stride" 7
#     "conv2d" "filter_count" 8 "kernel_size" 9 "stride" 10
#     """,
#     )
# )
# @example(
#     mesa.LayersTestData(
#         layers=(
#             gl.Conv2D("Conv2D_0", 1, 2, 3),
#             gl.make_fork("fork_0"),
#             gl.make_merge("merge_0"),
#             gl.Conv2D("Conv2D_1", 4, 5, 6),
#             gl.make_fork("fork_1"),
#             gl.make_merge("merge_1"),
#             gl.Conv2D("Conv2D_2", 7, 8, 9),
#             gl.Conv2D("Conv2D_3", 10, 11, 12),
#             gl.Conv2D("Conv2D_4", 13, 14, 15),
#             gl.make_merge("merge_2"),
#             gl.Conv2D("Conv2D_5", 16, 17, 18),
#         ),
#         mesagrammar_string="""
#     "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
#     "fork"

#     "merge"
#     "conv2d" "filter_count" 4 "kernel_size" 5 "stride" 6
#     "fork"

#     "merge"
#     "conv2d" "filter_count" 7 "kernel_size" 8 "stride" 9

#     "conv2d" "filter_count" 10 "kernel_size" 11 "stride" 12
#     "conv2d" "filter_count" 13 "kernel_size" 14"stride" 15
#     "merge"
#     "conv2d" "filter_count" 16 "kernel_size" 17 "stride" 18
#     """,
#     )
# )
# def test_conv2d_backbone(backbone: mesa.LayersTestData) -> None:
#     """Can parse a backbone of Conv2D with merge and fork points."""
#     actual = bb.parse(backbone.mesagrammar_string)
#     expected = bb.Backbone(backbone.layers)
#     assert actual == expected


# @given(mesa.backbone_grammar_layer())
# def test_parse_backbone(backbone: mesa.LayersTestData) -> None:
#     """Can parse a backbone with any layer, possibly marked."""
#     actual = bb.parse(backbone.mesagrammar_string)
#     expected = bb.Backbone(backbone.layers)
#     assert actual == expected


@given(test_data=mesa.add_dummy_optimizer_to(mesa.conv2ds()))
def test_parse_conv2d(test_data: mesa.LayersTestData) -> None:
    """Can parse conv2d."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=mesa.add_dummy_optimizer_to(mesa.max_pool2ds()))
def test_parse_max_pool2d(test_data: mesa.LayersTestData) -> None:
    """Can parse max pool2d."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=mesa.add_dummy_optimizer_to(mesa.avg_pool2ds()))
def test_parse_avg_pool2d(test_data: mesa.LayersTestData) -> None:
    """Can parse avg pool2d."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=mesa.add_dummy_optimizer_to(mesa.batchnorms()))
def test_parse_batchnorm(test_data: mesa.LayersTestData) -> None:
    """Can parse batchnorm."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=mesa.add_dummy_optimizer_to(mesa.relus()))
def test_parse_relu(test_data: mesa.LayersTestData) -> None:
    """Can parse relu."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=mesa.add_dummy_optimizer_to(mesa.gelus()))
def test_parse_gelu(test_data: mesa.LayersTestData) -> None:
    """Can parse gelu."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual


@given(test_data=mesa.add_dummy_optimizer_to(mesa.swishs()))
def test_parse_swish(test_data: mesa.LayersTestData) -> None:
    """Can parse swish."""
    actual = bb.parse(test_data.token_stream)
    expected = bb.Backbone(test_data.parsed)
    assert expected == actual
