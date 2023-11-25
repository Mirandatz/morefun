import attrs
from hypothesis import given

import morefun.grammars.backbones as bb
import morefun.name_generator as ng
import morefun.neural_networks.connections as conn
import morefun.neural_networks.layers as gl
import morefun.tests.strategies.layers as ls


@given(shapes=ls.shape_pairs())
def test_downsampling(shapes: ls.ShapePair) -> None:
    """Can downsample to exact fraction target output shape."""
    input_shape = shapes.bigger
    output_shape = shapes.smaller

    source = gl.ConnectedBatchNorm(
        input_layer=gl.Input(input_shape),
        params=gl.BatchNorm(name="bn"),
    )
    shortcut = conn.downsampling_shortcut(source, output_shape, name="whatever")

    assert shortcut.output_shape == output_shape


@given(shapes=ls.shape_pairs())
def test_upsampling(shapes: ls.ShapePair) -> None:
    """Can upsample to exact fraction target output shape."""
    input_shape = shapes.smaller
    output_shape = shapes.bigger

    source = gl.ConnectedBatchNorm(
        input_layer=gl.Input(input_shape), params=gl.BatchNorm(name="bn")
    )
    shortcut = conn.upsampling_shortcut(source, output_shape, name="whatever")

    assert shortcut.output_shape == output_shape


@given(shapes=ls.shape_pairs())
def test_downsampling_shortcut(shapes: ls.ShapePair) -> None:
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


@given(shapes=ls.shape_pairs())
def test_upsampling_shortcut(shapes: ls.ShapePair) -> None:
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


@given(shape=ls.shapes())
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


@given(shapes=ls.shape_pairs(same_depth=True))
def test_merge_downsample_add(shapes: ls.ShapePair) -> None:
    """Can add-merge using a downsample."""
    input_layer = gl.Input(shapes.bigger)

    source0 = gl.ConnectedBatchNorm(
        input_layer=input_layer,
        params=gl.BatchNorm(name="bn"),
    )

    source1 = gl.ConnectedMaxPool2D(
        input_layer=input_layer,
        params=gl.MaxPool2D("maxpool", pool_size=1, stride=shapes.ratio),
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


@given(shapes=ls.shape_pairs(same_depth=True))
def test_merge_downsample_concat(shapes: ls.ShapePair) -> None:
    """Concatenating sources sums the depths when downsampling preserving the smallest (width, height)."""
    input_layer = gl.Input(shapes.bigger)
    expected = attrs.evolve(shapes.smaller, depth=shapes.smaller.depth * 2)

    source0 = gl.ConnectedBatchNorm(
        input_layer=input_layer,
        params=gl.BatchNorm(name="bn"),
    )

    source1 = gl.ConnectedMaxPool2D(
        input_layer=input_layer,
        params=gl.MaxPool2D("maxpool", pool_size=1, stride=shapes.ratio),
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


@given(shapes=ls.shape_pairs(same_depth=True))
def test_merge_upsample_concat(shapes: ls.ShapePair) -> None:
    """Concatenating sources adds the depths when upsampling preserving the biggest (width, height)."""
    input_layer = gl.Input(shapes.bigger)
    expected = attrs.evolve(shapes.bigger, depth=shapes.bigger.depth * 2)

    source0 = gl.ConnectedBatchNorm(
        input_layer=input_layer,
        params=gl.BatchNorm(name="bn"),
    )

    source1 = gl.ConnectedMaxPool2D(
        input_layer=input_layer,
        params=gl.MaxPool2D("maxpool", pool_size=1, stride=shapes.ratio),
    )

    name_gen = ng.NameGenerator()

    merge = conn.make_merge(
        sources=[source0, source1],
        reshape_strategy=conn.ReshapeStrategy.UPSAMPLE,
        merge_strategy=conn.MergeStrategy.CONCAT,
        name_gen=name_gen,
    )

    assert isinstance(merge, gl.ConnectedConcatenate)
    assert len(merge.input_layers) == 2
    assert source0 in merge.input_layers
    assert source1 not in merge.input_layers  # shortcut instead
    assert merge.output_shape == expected


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
    candidates = [gl.Shape(height=2, width=1, depth=3)]

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
    # setup
    backbone = bb.Backbone(
        (gl.BatchNorm("bn0"), gl.make_fork("f0"), gl.make_merge("m0")),
    )

    merge_params = (
        conn.MergeParameters(
            forks_mask=(True,),
            merge_strategy=conn.MergeStrategy.ADD,
            reshape_strategy=conn.ReshapeStrategy.DOWNSAMPLE,
        ),
    )
    schema = conn.ConnectionsSchema(merge_params)

    # test
    input = gl.make_input(1, 2, 3)
    output_layer = conn.connect_backbone(backbone, schema, input_layer=input)

    assert isinstance(output_layer, gl.ConnectedBatchNorm)
    assert isinstance(output_layer.input_layer, gl.Input)
    assert output_layer.input_layer == input
