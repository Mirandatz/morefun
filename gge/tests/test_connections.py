import gge.backbones as bb
import gge.connections as conn
import gge.layers as gl
import gge.name_generator as ng


def test_downsampling() -> None:
    source = gl.ConnectedBatchNorm(
        input_layer=gl.Input(gl.Shape(10, 10, 3)),
        params=gl.BatchNorm(name="bn"),
    )
    target = gl.Shape(5, 5, 19)
    shortcut = conn.downsampling_shortcut(source, target, name="whatever")
    assert shortcut.output_shape == target


def test_upsampling() -> None:
    source = gl.ConnectedBatchNorm(
        input_layer=gl.Input(gl.Shape(10, 10, 3)),
        params=gl.BatchNorm(name="bn"),
    )
    target = gl.Shape(20, 20, 7)
    shortcut = conn.upsampling_shortcut(source, target, name="whatever")
    assert shortcut.output_shape == target


def test_shortcut() -> None:
    source = gl.ConnectedBatchNorm(
        input_layer=gl.Input(gl.Shape(10, 10, 3)),
        params=gl.BatchNorm(name="bn"),
    )
    name_gen = ng.NameGenerator()

    target = gl.Shape(1, 1, 7)
    shortcut = conn.make_shortcut(
        source, target, mode=conn.ReshapeStrategy.DOWNSAMPLE, name_gen=name_gen
    )
    assert shortcut.output_shape == target

    target = gl.Shape(100, 100, 3)
    shortcut = conn.make_shortcut(
        source, target, mode=conn.ReshapeStrategy.UPSAMPLE, name_gen=name_gen
    )
    assert shortcut.output_shape == target


def test_shortcut_identity() -> None:
    source = gl.ConnectedBatchNorm(
        input_layer=gl.Input(gl.Shape(10, 10, 3)),
        params=gl.BatchNorm(name="bn"),
    )
    name_gen = ng.NameGenerator()

    target = gl.Shape(10, 10, 3)
    shortcut = conn.make_shortcut(
        source, target, mode=conn.ReshapeStrategy.DOWNSAMPLE, name_gen=name_gen
    )
    assert source == shortcut

    target = gl.Shape(10, 10, 3)
    shortcut = conn.make_shortcut(
        source, target, mode=conn.ReshapeStrategy.UPSAMPLE, name_gen=name_gen
    )
    assert source == shortcut


def test_merge_downsample_add() -> None:
    input_layer = gl.Input(gl.Shape(10, 10, 3))

    source0 = gl.ConnectedBatchNorm(
        input_layer=input_layer,
        params=gl.BatchNorm(name="bn"),
    )

    source1 = gl.ConnectedPool2D(
        input_layer=input_layer,
        params=gl.Pool2D("maxpool", gl.PoolType.MAX_POOLING, stride=2),
    )

    name_gen = ng.NameGenerator()

    add = conn.make_merge(
        sources=[source0, source1],
        reshape_strategy=conn.ReshapeStrategy.DOWNSAMPLE,
        merge_strategy=conn.MergeStrategy.ADD,
        name_gen=name_gen,
    )

    assert isinstance(add, gl.ConnectedAdd)
    assert add.output_shape == gl.Shape(5, 5, 3)


def test_merge_downsample_concat() -> None:
    input_layer = gl.Input(gl.Shape(10, 10, 3))

    source0 = gl.ConnectedBatchNorm(
        input_layer=input_layer,
        params=gl.BatchNorm(name="bn"),
    )

    source1 = gl.ConnectedPool2D(
        input_layer=input_layer,
        params=gl.Pool2D("maxpool", gl.PoolType.MAX_POOLING, stride=2),
    )

    name_gen = ng.NameGenerator()

    add = conn.make_merge(
        sources=[source0, source1],
        reshape_strategy=conn.ReshapeStrategy.DOWNSAMPLE,
        merge_strategy=conn.MergeStrategy.CONCAT,
        name_gen=name_gen,
    )

    assert isinstance(add, gl.ConnectedConcatenate)
    assert add.output_shape == gl.Shape(5, 5, 6)


def test_merge_upsample_concat() -> None:
    input_layer = gl.Input(gl.Shape(10, 10, 3))

    source0 = gl.ConnectedBatchNorm(
        input_layer=input_layer,
        params=gl.BatchNorm(name="bn"),
    )

    source1 = gl.ConnectedPool2D(
        input_layer=input_layer,
        params=gl.Pool2D("maxpool", gl.PoolType.MAX_POOLING, stride=2),
    )

    name_gen = ng.NameGenerator()

    add = conn.make_merge(
        sources=[source0, source1],
        reshape_strategy=conn.ReshapeStrategy.UPSAMPLE,
        merge_strategy=conn.MergeStrategy.CONCAT,
        name_gen=name_gen,
    )

    assert isinstance(add, gl.ConnectedConcatenate)
    assert add.output_shape == gl.Shape(10, 10, 6)


def test_no_fork() -> None:
    layers = (gl.Conv2D("whatever", 1, 2, 3),)
    backbone = bb.Backbone(layers)
    actual = conn.collect_fork_sources(backbone)
    assert [] == actual


def test_one_fork() -> None:
    layers = (
        gl.Conv2D("c0", 1, 2, 3),
        gl.make_fork("fork"),
        gl.Conv2D("c1", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.collect_fork_sources(backbone)
    assert [backbone.layers[0]] == actual


def test_many_fork() -> None:
    layers = (
        gl.Conv2D("c0", 1, 2, 3),
        gl.make_fork("fork"),
        gl.Conv2D("c1", 1, 2, 3),
        gl.Conv2D("c2", 1, 2, 3),
        gl.Conv2D("c3", 1, 2, 3),
        gl.make_fork("bork"),
        gl.Conv2D("c4", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.collect_fork_sources(backbone)
    assert [layers[0], layers[4]] == actual


def test_no_fork_no_merge() -> None:
    layers = (gl.Conv2D("conv_0", 1, 2, 3),)
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected: tuple[int, ...] = tuple()
    assert expected == actual


def test_one_fork_no_merge() -> None:
    layers = (
        gl.Conv2D("conv_0", 1, 2, 3),
        gl.make_fork("fork_0"),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected: tuple[int, ...] = tuple()
    assert expected == actual


def test_no_fork_one_merge() -> None:
    layers = (
        gl.make_merge("merge_0"),
        gl.Conv2D("conv_0", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (0,)
    assert expected == actual


def test_many_fork_one_merge() -> None:
    layers = (
        gl.Conv2D("conv_0", 1, 2, 3),
        gl.make_fork("fork_0"),
        gl.Conv2D("conv_1", 1, 2, 3),
        gl.make_fork("fork_1"),
        gl.Conv2D("conv_2", 1, 2, 3),
        gl.make_merge("merge_0"),
        gl.Conv2D("conv_3", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (2,)
    assert expected == actual


def test_one_fork_many_merge() -> None:
    layers = (
        gl.Conv2D("conv_0", 1, 2, 3),
        gl.make_fork("fork_0"),
        gl.Conv2D("conv_1", 1, 2, 3),
        gl.make_merge("merge_0"),
        gl.Conv2D("conv_2", 1, 2, 3),
        gl.make_merge("merge_1"),
        gl.Conv2D("conv_3", 1, 2, 3),
        gl.make_merge("merge_2"),
        gl.Conv2D("conv_4", 1, 2, 3),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (1, 1, 1)
    assert expected == actual


def test_many_fork_many_merge() -> None:
    layers = (
        gl.Conv2D("conv_0", 1, 2, 3),
        gl.make_fork("fork_0"),
        gl.Conv2D("conv_1", 1, 2, 3),
        gl.make_fork("fork_1"),
        gl.Conv2D("conv_2", 1, 2, 3),
        gl.make_merge("merge_0"),
        gl.Conv2D("conv_3", 1, 2, 3),
        gl.make_merge("merge_1"),
        gl.Conv2D("conv_4", 1, 2, 3),
        gl.make_fork("fork_2"),
        gl.Conv2D("conv_5", 1, 2, 3),
        gl.make_merge("merge_2"),
    )
    backbone = bb.Backbone(layers)
    actual = conn.extract_forks_masks_lengths(backbone)
    expected = (2, 2, 3)
    assert expected == actual


def test_merge_parameters_constructor_empty_forks_mask() -> None:
    _ = conn.MergeParameters(
        forks_mask=tuple(),
        merge_strategy=conn.MergeStrategy.ADD,
        reshape_strategy=conn.ReshapeStrategy.DOWNSAMPLE,
    )
