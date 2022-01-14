import dataclasses

from hypothesis import given

import gge.backbones as bb
import gge.connections as conn
import gge.layers as gl
import gge.name_generator as ng
import gge.tests.strategies as gge_hs


@given(shapes=gge_hs.same_aspect_shape_pair())
def test_downsampling(shapes: gge_hs.ShapePair) -> None:
    """Can downsample to exact fraction target output shape."""
    input_shape = shapes.bigger
    output_shape = shapes.smaller

    source = gl.ConnectedBatchNorm(
        input_layer=gl.Input(input_shape),
        params=gl.BatchNorm(name="bn"),
    )
    shortcut = conn.downsampling_shortcut(source, output_shape, name="whatever")

    assert shortcut.output_shape == output_shape


@given(shapes=gge_hs.same_aspect_shape_pair())
def test_upsampling(shapes: gge_hs.ShapePair) -> None:
    """Can upsample to exact fraction target output shape."""
    input_shape = shapes.smaller
    output_shape = shapes.bigger

    source = gl.ConnectedBatchNorm(
        input_layer=gl.Input(input_shape), params=gl.BatchNorm(name="bn")
    )
    shortcut = conn.upsampling_shortcut(source, output_shape, name="whatever")

    assert shortcut.output_shape == output_shape


@given(shapes=gge_hs.same_aspect_shape_pair())
def test_downsampling_shortcut(shapes: gge_hs.ShapePair) -> None:
    """Can downsample to exact fraction target output shape from enum."""
    input_shape = shapes.bigger
    output_shape = shapes.smaller
    name_gen = ng.NameGenerator()

    source = gl.ConnectedBatchNorm(
        input_layer=gl.Input(input_shape), params=gl.BatchNorm(name="bn")
    )
    shortcut = conn.make_shortcut(
        source, output_shape, mode=conn.ReshapeStrategy.DOWNSAMPLE, name_gen=name_gen
    )

    assert shortcut.output_shape == output_shape


@given(shapes=gge_hs.same_aspect_shape_pair())
def test_upsampling_shortcut(shapes: gge_hs.ShapePair) -> None:
    """Can upsample to exact fraction target output shape from enum."""
    input_shape = shapes.smaller
    output_shape = shapes.bigger
    name_gen = ng.NameGenerator()

    source = gl.ConnectedBatchNorm(
        input_layer=gl.Input(input_shape), params=gl.BatchNorm(name="bn")
    )
    shortcut = conn.make_shortcut(
        source, output_shape, mode=conn.ReshapeStrategy.UPSAMPLE, name_gen=name_gen
    )

    assert shortcut.output_shape == output_shape


@given(shape=gge_hs.shape())
def test_shortcut_identity(shape: gl.Shape) -> None:
    """Making a shortcut with same shape returns the same layer regardless of method."""
    source = gl.ConnectedBatchNorm(
        input_layer=gl.Input(shape),
        params=gl.BatchNorm(name="bn"),
    )
    name_gen = ng.NameGenerator()

    downsample_shortcut = conn.make_shortcut(
        source, shape, mode=conn.ReshapeStrategy.DOWNSAMPLE, name_gen=name_gen
    )
    upsample_shortcut = conn.make_shortcut(
        source, shape, mode=conn.ReshapeStrategy.UPSAMPLE, name_gen=name_gen
    )

    assert source == downsample_shortcut
    assert source == upsample_shortcut


@given(shapes=gge_hs.same_aspect_shape_pair(same_depth=True))
def test_merge_downsample_add(shapes: gge_hs.ShapePair) -> None:
    """Can add-merge using a downsample."""
    input_layer = gl.Input(shapes.bigger)

    source0 = gl.ConnectedBatchNorm(
        input_layer=input_layer,
        params=gl.BatchNorm(name="bn"),
    )

    source1 = gl.ConnectedPool2D(
        input_layer=input_layer,
        params=gl.Pool2D("maxpool", gl.PoolType.MAX_POOLING, stride=shapes.ratio),
    )

    name_gen = ng.NameGenerator()

    add = conn.make_merge(
        sources=[source0, source1],
        reshape_strategy=conn.ReshapeStrategy.DOWNSAMPLE,
        merge_strategy=conn.MergeStrategy.ADD,
        name_gen=name_gen,
    )

    assert isinstance(add, gl.ConnectedAdd)
    assert add.output_shape == shapes.smaller


@given(shapes=gge_hs.same_aspect_shape_pair(same_depth=True))
def test_merge_downsample_concat(shapes: gge_hs.ShapePair) -> None:
    """Concatenating sources double the depth."""
    input_layer = gl.Input(shapes.bigger)
    expected = dataclasses.replace(shapes.smaller, depth=shapes.smaller.depth * 2)

    source0 = gl.ConnectedBatchNorm(
        input_layer=input_layer,
        params=gl.BatchNorm(name="bn"),
    )

    source1 = gl.ConnectedPool2D(
        input_layer=input_layer,
        params=gl.Pool2D("maxpool", gl.PoolType.MAX_POOLING, stride=shapes.ratio),
    )

    name_gen = ng.NameGenerator()

    add = conn.make_merge(
        sources=[source0, source1],
        reshape_strategy=conn.ReshapeStrategy.DOWNSAMPLE,
        merge_strategy=conn.MergeStrategy.CONCAT,
        name_gen=name_gen,
    )

    assert isinstance(add, gl.ConnectedConcatenate)
    assert add.output_shape == expected


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


def test_select_target_shape_single_candidate() -> None:
    candidates = [gl.Shape(1, 2, 3)]

    chosen_upsample = conn.select_target_shape(
        candidates,
        conn.ReshapeStrategy.UPSAMPLE,
    )

    chosen_downsample = conn.select_target_shape(
        candidates,
        conn.ReshapeStrategy.DOWNSAMPLE,
    )

    assert candidates[0] == chosen_downsample
    assert candidates[0] == chosen_upsample


def test_connect_backbone_with_duplicate_merge_entries() -> None:
    layers = gl.BatchNorm("bn0"), gl.make_fork("f0"), gl.make_merge("m0")
    backbone = bb.Backbone(layers)
    merge_params = (
        conn.MergeParameters(
            forks_mask=(True,),
            merge_strategy=conn.MergeStrategy.ADD,
            reshape_strategy=conn.ReshapeStrategy.DOWNSAMPLE,
        ),
    )
    schema = conn.ConnectionsSchema(merge_params)

    output = conn.connect_backbone(
        backbone,
        schema,
        input_layer=gl.make_input(1, 2, 3),
    )

    assert isinstance(output, gl.ConnectedAdd)

    (input,) = output.input_layers
    assert isinstance(input, gl.ConnectedBatchNorm)
