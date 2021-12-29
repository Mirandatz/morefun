import gge.backbones as bb
import gge.connections as conn
import gge.layers as gl


def test_no_fork_no_merge() -> None:
    layers = (gl.Conv2D("conv_0", 1, 2, 3),)
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected: tuple[int, ...] = tuple()
    assert expected == actual


def test_one_fork_no_merge() -> None:
    layers = (
        gl.Conv2D("conv_0", 1, 2, 3),
        gl.Fork("fork_0"),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected: tuple[int, ...] = tuple()
    assert expected == actual


def test_no_fork_one_merge() -> None:
    layers = (
        gl.Merge("merge_0"),
        gl.Conv2D("conv_0", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (0,)
    assert expected == actual


def test_many_fork_one_merge() -> None:
    layers = (
        gl.Conv2D("conv_0", 1, 2, 3),
        gl.Fork("fork_0"),
        gl.Conv2D("conv_1", 1, 2, 3),
        gl.Fork("fork_1"),
        gl.Conv2D("conv_2", 1, 2, 3),
        gl.Merge("merge_0"),
        gl.Conv2D("conv_3", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (2,)
    assert expected == actual


def test_one_fork_many_merge() -> None:
    layers = (
        gl.Conv2D("conv_0", 1, 2, 3),
        gl.Fork("fork_0"),
        gl.Conv2D("conv_1", 1, 2, 3),
        gl.Merge("merge_0"),
        gl.Conv2D("conv_2", 1, 2, 3),
        gl.Merge("merge_1"),
        gl.Conv2D("conv_3", 1, 2, 3),
        gl.Merge("merge_2"),
        gl.Conv2D("conv_4", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (1, 1, 1)
    assert expected == actual


def test_many_fork_many_merge() -> None:
    layers = (
        gl.Conv2D("conv_0", 1, 2, 3),
        gl.Fork("fork_0"),
        gl.Conv2D("conv_1", 1, 2, 3),
        gl.Fork("fork_1"),
        gl.Conv2D("conv_2", 1, 2, 3),
        gl.Merge("merge_0"),
        gl.Conv2D("conv_3", 1, 2, 3),
        gl.Merge("merge_1"),
        gl.Conv2D("conv_4", 1, 2, 3),
        gl.Fork("fork_2"),
        gl.Conv2D("conv_5", 1, 2, 3),
        gl.Merge("merge_2"),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (2, 2, 3)
    assert expected == actual
