import gge.backbones as bb
import gge.connections as conn


def test_no_fork() -> None:
    layers = (bb.Conv2DLayer("whatever", 1, 2, 3),)
    backbone = bb.Backbone(layers)
    actual = conn.collect_sources(backbone)
    assert [] == actual


def test_one_fork() -> None:
    layers = (
        bb.Conv2DLayer("c0", 1, 2, 3),
        bb.Fork("fork"),
        bb.Conv2DLayer("c1", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.collect_sources(backbone)
    assert [backbone.layers[0]] == actual


def test_many_fork() -> None:
    layers = (
        bb.Conv2DLayer("c0", 1, 2, 3),
        bb.Fork("fork"),
        bb.Conv2DLayer("c1", 1, 2, 3),
        bb.Conv2DLayer("c2", 1, 2, 3),
        bb.Conv2DLayer("c3", 1, 2, 3),
        bb.Fork("bork"),
        bb.Conv2DLayer("c4", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.collect_sources(backbone)
    assert [layers[0], layers[4]] == actual
