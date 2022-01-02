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
