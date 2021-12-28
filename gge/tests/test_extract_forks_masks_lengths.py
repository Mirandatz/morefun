import gge.backbones as bb
import gge.connections as conn


def test_0fork_0merge() -> None:
    layers = (bb.Conv2DLayer("conv_0", 1, 2, 3),)
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected: tuple[int, ...] = tuple()
    assert expected == actual


def test_1fork_0merge() -> None:
    layers = (
        bb.Conv2DLayer("conv_0", 1, 2, 3),
        bb.Fork("fork_0"),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected: tuple[int, ...] = tuple()
    assert expected == actual


def test_0fork_1merge() -> None:
    layers = (
        bb.Merge("merge_0"),
        bb.Conv2DLayer("conv_0", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (0,)
    assert expected == actual


def test_nfork_1merge() -> None:
    layers = (
        bb.Conv2DLayer("conv_0", 1, 2, 3),
        bb.Fork("fork_0"),
        bb.Conv2DLayer("conv_1", 1, 2, 3),
        bb.Fork("fork_1"),
        bb.Conv2DLayer("conv_2", 1, 2, 3),
        bb.Merge("merge_0"),
        bb.Conv2DLayer("conv_3", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (2,)
    assert expected == actual


def test_1fork_nmerge() -> None:
    layers = (
        bb.Conv2DLayer("conv_0", 1, 2, 3),
        bb.Fork("fork_0"),
        bb.Conv2DLayer("conv_1", 1, 2, 3),
        bb.Merge("merge_0"),
        bb.Conv2DLayer("conv_2", 1, 2, 3),
        bb.Merge("merge_1"),
        bb.Conv2DLayer("conv_3", 1, 2, 3),
        bb.Merge("merge_2"),
        bb.Conv2DLayer("conv_4", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (1, 1, 1)
    assert expected == actual


def test_nfork_nmerge() -> None:
    layers = (
        bb.Conv2DLayer("conv_0", 1, 2, 3),
        bb.Fork("fork_0"),
        bb.Conv2DLayer("conv_1", 1, 2, 3),
        bb.Fork("fork_1"),
        bb.Conv2DLayer("conv_2", 1, 2, 3),
        bb.Merge("merge_0"),
        bb.Conv2DLayer("conv_3", 1, 2, 3),
        bb.Merge("merge_1"),
        bb.Conv2DLayer("conv_4", 1, 2, 3),
        bb.Fork("fork_2"),
        bb.Conv2DLayer("conv_5", 1, 2, 3),
        bb.Merge("merge_2"),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (2, 2, 3)
    assert expected == actual
