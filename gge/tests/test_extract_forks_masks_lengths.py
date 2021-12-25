import gge.backbones as bb
import gge.connections as conn


def test_0fork_0merge() -> None:
    layers = (bb.DenseLayer("dense_0", 1),)
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected: tuple[int, ...] = tuple()
    assert expected == actual


def test_1fork_0merge() -> None:
    layers = (
        bb.DenseLayer("dense_0", 1),
        bb.Fork("fork_0"),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected: tuple[int, ...] = tuple()
    assert expected == actual


def test_0fork_1merge() -> None:
    layers = (
        bb.Merge("merge_0"),
        bb.DenseLayer("dense_0", 2),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (0,)
    assert expected == actual


def test_nfork_1merge() -> None:
    layers = (
        bb.DenseLayer("dense_0", 1),
        bb.Fork("fork_0"),
        bb.DenseLayer("dense_1", 2),
        bb.Fork("fork_1"),
        bb.DenseLayer("dense_2", 3),
        bb.Merge("merge_0"),
        bb.DenseLayer("dense_3", 4),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (2,)
    assert expected == actual


def test_1fork_nmerge() -> None:
    layers = (
        bb.DenseLayer("dense_0", 1),
        bb.Fork("fork_0"),
        bb.DenseLayer("dense_1", 2),
        bb.Merge("merge_0"),
        bb.DenseLayer("dense_2", 3),
        bb.Merge("merge_1"),
        bb.DenseLayer("dense_3", 4),
        bb.Merge("merge_2"),
        bb.DenseLayer("dense_4", 5),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (1, 1, 1)
    assert expected == actual


def test_nfork_nmerge() -> None:
    layers = (
        bb.DenseLayer("dense_0", 1),
        bb.Fork("fork_0"),
        bb.DenseLayer("dense_1", 2),
        bb.Fork("fork_1"),
        bb.DenseLayer("dense_2", 3),
        bb.Merge("merge_0"),
        bb.DenseLayer("dense_3", 4),
        bb.Merge("merge_1"),
        bb.DenseLayer("dense_4", 5),
        bb.Fork("fork_2"),
        bb.DenseLayer("dense_5", 6),
        bb.Merge("merge_2"),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (2, 2, 3)
    assert expected == actual
