import itertools

import gge.layers as gl


def test_same_aspect_ratio() -> None:
    shapes = [
        gl.Shape(4, 3, 19),
        gl.Shape(8, 6, 12),
        gl.Shape(4, 3, 19),
    ]

    for a, b in itertools.pairwise(shapes):
        assert a.aspect_ratio == b.aspect_ratio


def test_different_aspect_ratio() -> None:
    a = gl.Shape(1, 2, 3)
    b = gl.Shape(1, 3, 3)

    assert a.aspect_ratio != b.aspect_ratio
