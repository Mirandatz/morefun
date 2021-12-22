import gge.backbones as bb


def test_conv2d() -> None:
    tokenstream = """
    conv2d filter_count 1 kernel_size 2 stride 3
    """

    actual = bb.parse(tokenstream)
    layers = (bb.Conv2DLayer(filter_count=1, kernel_size=2, stride=3),)
    expected = bb.Backbone(layers)
    assert expected == actual


def test_dense() -> None:
    tokenstream = """
    dense 5
    """

    actual = bb.parse(tokenstream)

    layers = (bb.DenseLayer(5),)
    expected = bb.Backbone(layers)
    assert expected == actual


def test_dropout() -> None:
    tokenstream = """
    dropout 0.7
    """

    actual = bb.parse(tokenstream)

    layers = (bb.DropoutLayer(0.7),)
    expected = bb.Backbone(layers)
    assert expected == actual


def test_merge_() -> None:
    tokenstream = """
    merge dense 2
    """

    actual = bb.parse(tokenstream)

    layers = (bb.Merge(), bb.DenseLayer(2))
    expected = bb.Backbone(layers)
    assert expected == actual


def test_fork() -> None:
    tokenstream = """
    dense 1 fork
    """

    actual = bb.parse(tokenstream)

    layers = (bb.DenseLayer(1), bb.Fork())
    expected = bb.Backbone(layers)
    assert expected == actual


def test_merge_and_fork() -> None:
    tokenstream = """
    merge dense 5 fork
    """

    actual = bb.parse(tokenstream)

    layers = (bb.Merge(), bb.DenseLayer(5), bb.Fork())
    expected = bb.Backbone(layers)
    assert expected == actual


def test_simple_backbone() -> None:
    tokenstream = """
    conv2d filter_count 1 kernel_size 2 stride 3
    conv2d filter_count 4 kernel_size 5 stride 6
    dense 7
    dropout 0.8
    dense 9
    dropout 0.10
    """

    actual = bb.parse(tokenstream)

    layers = (
        bb.Conv2DLayer(1, 2, 3),
        bb.Conv2DLayer(4, 5, 6),
        bb.DenseLayer(7),
        bb.DropoutLayer(0.8),
        bb.DenseLayer(9),
        bb.DropoutLayer(0.10),
    )
    expected = bb.Backbone(layers)

    assert expected == actual


def test_complex_backbone() -> None:
    tokenstream = """
    conv2d filter_count 1 kernel_size 2 stride 3 fork
    merge conv2d filter_count 4 kernel_size 5 stride 6 fork
    merge conv2d filter_count 7 kernel_size 8 stride 9

    dense 10
    dropout 0.11 fork

    dense 12
    dropout 0.13 fork

    merge dense 14
    dropout 0.15

    """

    actual = bb.parse(tokenstream)

    a: list[bb.Layer] = [bb.Conv2DLayer(1, 2, 3), bb.Fork()]
    b: list[bb.Layer] = [bb.Merge(), bb.Conv2DLayer(4, 5, 6), bb.Fork()]
    c: list[bb.Layer] = [bb.Merge(), bb.Conv2DLayer(7, 8, 9)]
    d: list[bb.Layer] = [bb.DenseLayer(10), bb.DropoutLayer(0.11), bb.Fork()]
    e: list[bb.Layer] = [bb.DenseLayer(12), bb.DropoutLayer(0.13), bb.Fork()]
    f: list[bb.Layer] = [bb.Merge(), bb.DenseLayer(14), bb.DropoutLayer(0.15)]

    layers = tuple([*a, *b, *c, *d, *e, *f])
    expected = bb.Backbone(layers)

    assert expected == actual
