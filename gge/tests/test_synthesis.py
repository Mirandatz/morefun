import gge.synthesis as syn


def test_conv2d() -> None:
    tokenstream = """
    conv2d filter_count 1 kernel_size 2 stride 3
    """

    actual = syn.parse_tokenstream(tokenstream)
    layers = (syn.Conv2DLayer(filter_count=1, kernel_size=2, stride=3),)
    expected = syn.Backbone(layers)
    assert expected == actual


def test_dense() -> None:
    tokenstream = """
    dense 5
    """

    actual = syn.parse_tokenstream(tokenstream)

    layers = (syn.DenseLayer(5),)
    expected = syn.Backbone(layers)
    assert expected == actual


def test_dropout() -> None:
    tokenstream = """
    dropout 0.7
    """

    actual = syn.parse_tokenstream(tokenstream)

    layers = (syn.DropoutLayer(0.7),)
    expected = syn.Backbone(layers)
    assert expected == actual


def test_merge_() -> None:
    tokenstream = """
    merge dense 2
    """

    actual = syn.parse_tokenstream(tokenstream)

    layers = (syn.Merge(), syn.DenseLayer(2))
    expected = syn.Backbone(layers)
    assert expected == actual


def test_fork() -> None:
    tokenstream = """
    dense 1 fork
    """

    actual = syn.parse_tokenstream(tokenstream)

    layers = (syn.DenseLayer(1), syn.Fork())
    expected = syn.Backbone(layers)
    assert expected == actual


def test_merge_and_fork() -> None:
    tokenstream = """
    merge dense 5 fork
    """

    actual = syn.parse_tokenstream(tokenstream)

    layers = (syn.Merge(), syn.DenseLayer(5), syn.Fork())
    expected = syn.Backbone(layers)
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

    actual = syn.parse_tokenstream(tokenstream)

    layers = (
        syn.Conv2DLayer(1, 2, 3),
        syn.Conv2DLayer(4, 5, 6),
        syn.DenseLayer(7),
        syn.DropoutLayer(0.8),
        syn.DenseLayer(9),
        syn.DropoutLayer(0.10),
    )
    expected = syn.Backbone(layers)

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

    actual = syn.parse_tokenstream(tokenstream)

    a: list[syn.Layer] = [syn.Conv2DLayer(1, 2, 3), syn.Fork()]
    b: list[syn.Layer] = [syn.Merge(), syn.Conv2DLayer(4, 5, 6), syn.Fork()]
    c: list[syn.Layer] = [syn.Merge(), syn.Conv2DLayer(7, 8, 9)]
    d: list[syn.Layer] = [syn.DenseLayer(10), syn.DropoutLayer(0.11), syn.Fork()]
    e: list[syn.Layer] = [syn.DenseLayer(12), syn.DropoutLayer(0.13), syn.Fork()]
    f: list[syn.Layer] = [syn.Merge(), syn.DenseLayer(14), syn.DropoutLayer(0.15)]

    layers = tuple([*a, *b, *c, *d, *e, *f])
    expected = syn.Backbone(layers)

    assert expected == actual
