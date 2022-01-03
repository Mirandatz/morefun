import gge.layers as gl


def test_conv2d_output_depth_non_unity_stride() -> None:
    connected = gl.ConnectedConv2D(
        gl.Input(gl.Shape(1024, 1024, 3)),
        gl.Conv2D(name="a", filter_count=17, kernel_size=5, stride=4),
    )

    actual = connected.output_shape
    expected = gl.Shape(1024 // 4, 1024 // 4, 17)
    assert expected == actual


def test_conv2d_output_depth_unity_stride() -> None:
    connected = gl.ConnectedConv2D(
        gl.Input(gl.Shape(1024, 1024, 3)),
        gl.Conv2D(name="a", filter_count=17, kernel_size=5, stride=1),
    )

    actual = connected.output_shape
    expected = gl.Shape(1024, 1024, 17)
    assert expected == actual
