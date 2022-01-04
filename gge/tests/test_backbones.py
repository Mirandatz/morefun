import pytest

import gge.backbones as bb
import gge.layers as gl


def test_conv2d() -> None:
    tokenstream = """
    "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    """

    actual = bb.parse(tokenstream)
    layers = (gl.Conv2D("conv2d_0", 1, 2, 3),)
    expected = bb.Backbone(layers)
    assert expected == actual


def test_merge() -> None:
    tokenstream = """
    "merge" "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    """

    actual = bb.parse(tokenstream)
    layers = (
        gl.make_merge("merge_0"),
        gl.Conv2D("conv2d_0", 1, 2, 3),
    )
    expected = bb.Backbone(layers)
    assert expected == actual


def test_fork() -> None:
    tokenstream = """
    "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "fork"
    """

    actual = bb.parse(tokenstream)
    layers = (
        gl.Conv2D("conv2d_0", 1, 2, 3),
        gl.make_fork("fork_0"),
    )
    expected = bb.Backbone(layers)
    assert expected == actual


def test_merge_and_fork() -> None:
    tokenstream = """
    "merge" "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "fork"
    """

    actual = bb.parse(tokenstream)
    layers = (
        gl.make_merge("merge_0"),
        gl.Conv2D("conv2d_0", 1, 2, 3),
        gl.make_fork("fork_0"),
    )
    expected = bb.Backbone(layers)
    assert expected == actual


def test_simple_backbone() -> None:
    tokenstream = """
    "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    "conv2d" "filter_count" 5 "kernel_size" 6 "stride" 7
    "conv2d" "filter_count" 8 "kernel_size" 9 "stride" 10
    """

    actual = bb.parse(tokenstream)
    layers = (
        gl.Conv2D("conv2d_0", 1, 2, 3),
        gl.Conv2D("conv2d_1", 5, 6, 7),
        gl.Conv2D("conv2d_2", 8, 9, 10),
    )
    expected = bb.Backbone(layers)

    assert expected == actual


def test_complex_backbone() -> None:
    tokenstream = """
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
    """

    actual = bb.parse(tokenstream)
    layers = (
        gl.Conv2D("conv2d_0", 1, 2, 3),
        gl.make_fork("fork_0"),
        gl.make_merge("merge_0"),
        gl.Conv2D("conv2d_1", 4, 5, 6),
        gl.make_fork("fork_1"),
        gl.make_merge("merge_1"),
        gl.Conv2D("conv2d_2", 7, 8, 9),
        gl.Conv2D("conv2d_3", 10, 11, 12),
        gl.Conv2D("conv2d_4", 13, 14, 15),
        gl.make_merge("merge_2"),
        gl.Conv2D("conv2d_5", 16, 17, 18),
    )
    expected = bb.Backbone(layers)
    assert expected == actual


def test_standalone_batchnorm() -> None:
    tokenstream = """
    "batchnorm"
    """
    actual = bb.parse(tokenstream)
    layers = (gl.BatchNorm("batchnorm_0"),)
    expected = bb.Backbone(layers)
    assert expected == actual


def test_batchnorm_after_conv() -> None:
    tokenstream = """
    "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    "batchnorm"
    """
    actual = bb.parse(tokenstream)
    layers = (
        gl.Conv2D("conv2d_0", 1, 2, 3),
        gl.BatchNorm("batchnorm_0"),
    )
    expected = bb.Backbone(layers)
    assert expected == actual


@pytest.mark.parametrize(
    argnames=["text", "layer"],
    argvalues=[
        ('"max"', gl.Pool2D("pooling_layer_0", gl.PoolType.MAX_POOLING, 1)),
        ('"avg"', gl.Pool2D("pooling_layer_0", gl.PoolType.AVG_POOLING, 1)),
    ],
)
def test_pooling_layer_type(text: str, layer: gl.Pool2D) -> None:
    tokenstream = f"""
    "pool2d" {text} 1
    """
    actual = bb.parse(tokenstream)
    expected = bb.Backbone((layer,))
    assert expected == actual


@pytest.mark.parametrize(
    argnames=["text", "layer"],
    argvalues=[
        ("1", gl.Pool2D("pooling_layer_0", gl.PoolType.MAX_POOLING, 1)),
        ("2", gl.Pool2D("pooling_layer_0", gl.PoolType.MAX_POOLING, 2)),
    ],
)
def test_pooling_layer_stride(text: str, layer: gl.Pool2D) -> None:
    tokenstream = f"""
    "pool2d" "max" {text}
    """
    actual = bb.parse(tokenstream)
    expected = bb.Backbone((layer,))
    assert expected == actual
