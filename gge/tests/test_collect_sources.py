import gge.backbones as bb
import gge.connections as conn
import gge.layers as gl


def test_no_fork() -> None:
    layers = (gl.Conv2D("whatever", 1, 2, 3),)
    backbone = bb.Backbone(layers)
    actual = conn.collect_fork_sources(backbone)
    assert [] == actual


def test_one_fork() -> None:
    layers = (
        gl.Conv2D("c0", 1, 2, 3),
        gl.Fork("fork"),
        gl.Conv2D("c1", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.collect_fork_sources(backbone)
    assert [backbone.layers[0]] == actual


def test_many_fork() -> None:
    layers = (
        gl.Conv2D("c0", 1, 2, 3),
        gl.Fork("fork"),
        gl.Conv2D("c1", 1, 2, 3),
        gl.Conv2D("c2", 1, 2, 3),
        gl.Conv2D("c3", 1, 2, 3),
        gl.Fork("bork"),
        gl.Conv2D("c4", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.collect_fork_sources(backbone)
    assert [layers[0], layers[4]] == actual
