import gge.backbones as bb


def test_conv2d() -> None:
    tokenstream = """
    "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    """

    actual = bb.parse(tokenstream)
    layers = (bb.Conv2DLayer("conv2d_0", 1, 2, 3),)
    expected = bb.Backbone(layers)
    assert expected == actual


def test_dense() -> None:
    tokenstream = """
    "dense" 5
    """

    actual = bb.parse(tokenstream)
    layers = (bb.DenseLayer("dense_0", 5),)
    expected = bb.Backbone(layers)
    assert expected == actual


def test_dropout() -> None:
    tokenstream = """
    "dropout" 0.7
    """

    actual = bb.parse(tokenstream)
    layers = (bb.DropoutLayer("dropout_0", 0.7),)
    expected = bb.Backbone(layers)
    assert expected == actual


def test_merge_() -> None:
    tokenstream = """
    "merge" "dense" 2
    """

    actual = bb.parse(tokenstream)
    layers = (
        bb.Merge("merge_0"),
        bb.DenseLayer("dense_0", 2),
    )
    expected = bb.Backbone(layers)
    assert expected == actual


def test_fork() -> None:
    tokenstream = """
    "dense" 1 "fork"
    """

    actual = bb.parse(tokenstream)
    layers = (
        bb.DenseLayer("dense_0", 1),
        bb.Fork("fork_0"),
    )
    expected = bb.Backbone(layers)
    assert expected == actual


def test_merge_and_fork() -> None:
    tokenstream = """
    "merge" "dense" 5 "fork"
    """

    actual = bb.parse(tokenstream)
    layers = (
        bb.Merge("merge_0"),
        bb.DenseLayer("dense_0", 5),
        bb.Fork("fork_0"),
    )
    expected = bb.Backbone(layers)
    assert expected == actual


def test_simple_backbone() -> None:
    tokenstream = """
    "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    "conv2d" "filter_count" 4 "kernel_size" 5 "stride" 6
    "dense" 7
    "dropout" 0.8
    "dense" 9
    "dropout" 0.10
    """

    actual = bb.parse(tokenstream)
    layers = (
        bb.Conv2DLayer("conv2d_0", 1, 2, 3),
        bb.Conv2DLayer("conv2d_1", 4, 5, 6),
        bb.DenseLayer("dense_0", 7),
        bb.DropoutLayer("dropout_0", 0.8),
        bb.DenseLayer("dense_1", 9),
        bb.DropoutLayer("dropout_1", 0.10),
    )
    expected = bb.Backbone(layers)

    assert expected == actual


def test_complex_backbone() -> None:
    tokenstream = """
    "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    "fork"

    "merge"
    "conv2d" "filter_count" 4 "kernel_size" 5 "stride" 6
    "fork"

    "merge"
    "conv2d" "filter_count" 7 "kernel_size" 8 "stride" 9

    "dense" 10
    "dropout" 0.11 "fork"

    "dense" 12
    "dropout" 0.13 "fork"

    "merge"
    "dense" 14
    "dropout" 0.15
    """

    actual = bb.parse(tokenstream)
    layers = (
        bb.Conv2DLayer("conv2d_0", 1, 2, 3),
        bb.Fork("fork_0"),
        bb.Merge("merge_0"),
        bb.Conv2DLayer("conv2d_1", 4, 5, 6),
        bb.Fork("fork_1"),
        bb.Merge("merge_1"),
        bb.Conv2DLayer("conv2d_2", 7, 8, 9),
        bb.DenseLayer("dense_0", 10),
        bb.DropoutLayer("dropout_0", 0.11),
        bb.Fork("fork_2"),
        bb.DenseLayer("dense_1", 12),
        bb.DropoutLayer("dropout_1", 0.13),
        bb.Fork("fork_3"),
        bb.Merge("merge_2"),
        bb.DenseLayer("dense_2", 14),
        bb.DropoutLayer("dropout_2", 0.15),
    )
    expected = bb.Backbone(layers)
    assert expected == actual
