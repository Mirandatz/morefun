import gge.connections as conn
import gge.synthesis as syn


def test_0fork_0merge() -> None:
    layers = (syn.DenseLayer(1),)
    backbone = syn.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected: tuple[int, ...] = tuple()
    assert expected == actual


def test_1fork_0merge() -> None:
    layers = (syn.DenseLayer(1), syn.Fork())
    backbone = syn.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected: tuple[int, ...] = tuple()
    assert expected == actual


def test_0fork_1merge() -> None:
    layers = (syn.Merge(), syn.DenseLayer(2))
    backbone = syn.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (0,)
    assert expected == actual


def test_nfork_1merge() -> None:
    layers = (
        (syn.DenseLayer(1), syn.Fork())
        + (syn.DenseLayer(2), syn.Fork())
        + (syn.DenseLayer(3), syn.Merge())
        + (syn.DenseLayer(4),)
    )
    backbone = syn.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (2,)
    assert expected == actual


def test_1fork_nmerge() -> None:
    layers = (
        (syn.DenseLayer(1), syn.Fork())
        + (syn.DenseLayer(2), syn.Merge())
        + (syn.DenseLayer(3), syn.Merge())
        + (syn.DenseLayer(4), syn.Merge())
        + (syn.DenseLayer(5),)
    )
    backbone = syn.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (1, 1, 1)
    assert expected == actual


def test_nfork_nmerge() -> None:
    layers = (
        (syn.DenseLayer(1), syn.Fork())
        + (syn.DenseLayer(2), syn.Fork())
        + (syn.DenseLayer(3), syn.Merge())
        + (syn.DenseLayer(4), syn.Merge())
        + (syn.DenseLayer(5), syn.Fork())
        + (syn.DenseLayer(6), syn.Merge())
    )
    backbone = syn.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (2, 2, 3)
    assert expected == actual
