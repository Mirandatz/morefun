import gge.backbones as bb
import gge.connections as conn


def test_0fork_0merge() -> None:
    layers = (bb.DenseLayer(1),)
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected: tuple[int, ...] = tuple()
    assert expected == actual


def test_1fork_0merge() -> None:
    layers = (bb.DenseLayer(1), bb.Fork())
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected: tuple[int, ...] = tuple()
    assert expected == actual


def test_0fork_1merge() -> None:
    layers = (bb.Merge(), bb.DenseLayer(2))
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (0,)
    assert expected == actual


def test_nfork_1merge() -> None:
    layers = (
        (bb.DenseLayer(1), bb.Fork())
        + (bb.DenseLayer(2), bb.Fork())
        + (bb.DenseLayer(3), bb.Merge())
        + (bb.DenseLayer(4),)
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (2,)
    assert expected == actual


def test_1fork_nmerge() -> None:
    layers = (
        (bb.DenseLayer(1), bb.Fork())
        + (bb.DenseLayer(2), bb.Merge())
        + (bb.DenseLayer(3), bb.Merge())
        + (bb.DenseLayer(4), bb.Merge())
        + (bb.DenseLayer(5),)
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (1, 1, 1)
    assert expected == actual


def test_nfork_nmerge() -> None:
    layers = (
        (bb.DenseLayer(1), bb.Fork())
        + (bb.DenseLayer(2), bb.Fork())
        + (bb.DenseLayer(3), bb.Merge())
        + (bb.DenseLayer(4), bb.Merge())
        + (bb.DenseLayer(5), bb.Fork())
        + (bb.DenseLayer(6), bb.Merge())
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (2, 2, 3)
    assert expected == actual
