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

    assert isinstance(add, gl.Add)
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

    assert isinstance(add, gl.Concat)
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

    assert isinstance(add, gl.Concat)
    assert add.output_shape == gl.Shape(10, 10, 6)
