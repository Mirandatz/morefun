import gge.backbones as bb


def test_conv2d() -> None:
    tokenstream = """
    "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    """

    actual = bb.parse(tokenstream)
    layers = (bb.Conv2DLayer("conv2d_0", 1, 2, 3),)
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


def test_merge() -> None:
    tokenstream = """
    "merge" "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    """

    actual = bb.parse(tokenstream)
    layers = (
        bb.Merge("merge_0"),
        bb.Conv2DLayer("conv2d_0", 1, 2, 3),
    )
    expected = bb.Backbone(layers)
    assert expected == actual


def test_fork() -> None:
    tokenstream = """
    "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "fork"
    """

    actual = bb.parse(tokenstream)
    layers = (
        bb.Conv2DLayer("conv2d_0", 1, 2, 3),
        bb.Fork("fork_0"),
    )
    expected = bb.Backbone(layers)
    assert expected == actual


def test_merge_and_fork() -> None:
    tokenstream = """
    "merge" "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3 "fork"
    """

    actual = bb.parse(tokenstream)
    layers = (
        bb.Merge("merge_0"),
        bb.Conv2DLayer("conv2d_0", 1, 2, 3),
        bb.Fork("fork_0"),
    )
    expected = bb.Backbone(layers)
    assert expected == actual


def test_simple_backbone() -> None:
    tokenstream = """
    "conv2d" "filter_count" 1 "kernel_size" 2 "stride" 3
    "dropout" 0.4
    "conv2d" "filter_count" 5 "kernel_size" 6 "stride" 7
    "dropout" 0.8
    """

    actual = bb.parse(tokenstream)
    layers = (
        bb.Conv2DLayer("conv2d_0", 1, 2, 3),
        bb.DropoutLayer("dropout_0", 0.4),
        bb.Conv2DLayer("conv2d_1", 5, 6, 7),
        bb.DropoutLayer("dropout_1", 0.8),
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

    "conv2d" "filter_count" 10 "kernel_size" 11 "stride" 12
    "conv2d" "filter_count" 13 "kernel_size" 14"stride" 15
    "merge"
    "conv2d" "filter_count" 16 "kernel_size" 17 "stride" 18
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
        bb.Conv2DLayer("conv2d_3", 10, 11, 12),
        bb.Conv2DLayer("conv2d_4", 13, 14, 15),
        bb.Merge("merge_2"),
        bb.Conv2DLayer("conv2d_5", 16, 17, 18),
    )
    expected = bb.Backbone(layers)
    assert expected == actual
