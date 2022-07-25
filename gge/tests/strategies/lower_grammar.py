import attrs
import hypothesis.strategies as hs

import gge.layers as gl
import gge.optimizers as optim
from gge.name_generator import NameGenerator
from gge.tests.strategies import data_structures as tds

LowerGrammarParsingTestData = tds.ParsingTestData[tuple[gl.Layer, ...]]


@hs.composite
def random_flips(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.RandomFlip)
    mode = draw(
        hs.sampled_from(['"horizontal"', '"vertical"', '"horizontal_and_vertical"'])
    )
    layer = gl.RandomFlip(name, mode)
    tokenstream = f'"random_flip" {mode}'
    return LowerGrammarParsingTestData(tokenstream, (layer,))


@hs.composite
def random_rotations(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.RandomRotation)
    factor = draw(hs.floats(min_value=0, max_value=1))
    layer = gl.RandomRotation(name, factor)
    tokenstream = f'"random_rotation" {factor}'
    return LowerGrammarParsingTestData(tokenstream, (layer,))


@hs.composite
def resizings(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.Resizing)
    target_height = draw(hs.integers(min_value=1, max_value=9999))
    target_width = draw(hs.integers(min_value=1, max_value=9999))
    tokenstream = f'"resizing" "height" {target_height} "width" {target_width}'
    layer = gl.Resizing(
        name,
        height=target_height,
        width=target_width,
    )
    return LowerGrammarParsingTestData(tokenstream, (layer,))


@hs.composite
def random_crops(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.RandomCrop)
    height = draw(hs.integers(min_value=1, max_value=9999))
    width = draw(hs.integers(min_value=1, max_value=9999))
    layer = gl.RandomCrop(name, height=height, width=width)
    tokenstream = f'"random_crop" "height" {height} "width"  {width}'
    return LowerGrammarParsingTestData(tokenstream, (layer,))


@hs.composite
def convs(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.Conv2D)
    filter_count = draw(hs.integers(min_value=1))
    kernel_size = draw(hs.integers(min_value=1))
    stride = draw(hs.integers(min_value=1))
    layer = gl.Conv2D(name, filter_count, kernel_size, stride)
    tokenstream = f'"conv" "filter_count" {filter_count} "kernel_size" {kernel_size} "stride" {stride}'
    return LowerGrammarParsingTestData(tokenstream, (layer,))


@hs.composite
def maxpools(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.MaxPool2D)
    pool_size = draw(hs.integers(min_value=1, max_value=9999))
    stride = draw(hs.integers(min_value=1, max_value=9999))
    layer = gl.MaxPool2D(name, pool_size=pool_size, stride=stride)
    tokenstream = f'"maxpool" "pool_size" {pool_size} "stride" {stride}'
    return LowerGrammarParsingTestData(tokenstream, (layer,))


@hs.composite
def avgpools(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.AvgPool2D)
    pool_size = draw(hs.integers(min_value=1, max_value=9999))
    stride = draw(hs.integers(min_value=1, max_value=9999))
    layer = gl.AvgPool2D(name, pool_size=pool_size, stride=stride)
    tokenstream = f'"avgpool" "pool_size" {pool_size} "stride" {stride}'
    return LowerGrammarParsingTestData(tokenstream, (layer,))


def batchnorms(*, name_gen: NameGenerator | None = None) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.BatchNorm)
    return LowerGrammarParsingTestData(
        tokenstream='"batchnorm"',
        parsed=(gl.BatchNorm(name),),
    )


def relus(*, name_gen: NameGenerator | None = None) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.Relu)
    return LowerGrammarParsingTestData(
        tokenstream='"relu"',
        parsed=(gl.Relu(name),),
    )


def gelus(*, name_gen: NameGenerator | None = None) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.Gelu)
    return LowerGrammarParsingTestData(
        tokenstream='"gelu"',
        parsed=(gl.Gelu(name),),
    )


def swishs(*, name_gen: NameGenerator | None = None) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.Swish)
    return LowerGrammarParsingTestData(
        tokenstream='"swish"',
        parsed=(gl.Swish(name),),
    )


def prelus(*, name_gen: NameGenerator | None = None) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.Prelu)
    return LowerGrammarParsingTestData(
        tokenstream='"prelu"',
        parsed=(gl.Prelu(name),),
    )


def merge_marker(
    *, name_gen: NameGenerator | None = None
) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name("merge")
    return LowerGrammarParsingTestData(
        tokenstream='"merge"',
        parsed=(gl.make_merge(name),),
    )


def fork_marker(
    *, name_gen: NameGenerator | None = None
) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name("fork")
    return LowerGrammarParsingTestData(
        tokenstream='"fork"',
        parsed=(gl.make_fork(name),),
    )


@hs.composite
def add_dummy_optimizer_suffix_to(
    draw: hs.DrawFn,
    strat: hs.SearchStrategy[LowerGrammarParsingTestData],
) -> LowerGrammarParsingTestData:
    test_data = draw(strat)
    dummy_optimizer = '"sgd" "learning_rate" 0.1 "momentum" 0.05 "nesterov" false'
    updated_tokenstream = test_data.tokenstream + dummy_optimizer
    return attrs.evolve(test_data, tokenstream=updated_tokenstream)


@attrs.frozen
class OptimizerTestData:
    tokenstream: str
    parsed: optim.Optimizer


@hs.composite
def sgds(draw: hs.DrawFn) -> OptimizerTestData:
    learning_rate = draw(hs.floats(min_value=0, max_value=9, exclude_min=True))
    momentum = draw(hs.floats(min_value=0, max_value=1))
    nesterov = draw(hs.booleans())
    sgd = optim.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
    )
    tokenstream = f'"sgd" "learning_rate" {learning_rate} "momentum" {momentum} "nesterov" {str(nesterov).lower()}'
    return OptimizerTestData(tokenstream, sgd)


@hs.composite
def adams(draw: hs.DrawFn) -> OptimizerTestData:
    learning_rate = draw(hs.floats(min_value=0, max_value=9, exclude_min=True))
    beta1 = draw(hs.floats(min_value=0, max_value=9, exclude_min=True))
    beta2 = draw(hs.floats(min_value=0, max_value=9, exclude_min=True))
    epsilon = draw(hs.floats(min_value=0, max_value=9, exclude_min=True))
    amsgrad = draw(hs.booleans())
    adam = optim.Adam(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        amsgrad=amsgrad,
    )
    tokenstream = (
        f'"adam" "learning_rate" {learning_rate}'
        f'"beta1" {beta1}'
        f'"beta2" {beta2}'
        f'"epsilon" {epsilon}'
        f'"amsgrad" {str(amsgrad).lower()}'
    )
    return OptimizerTestData(tokenstream=tokenstream, parsed=adam)


@hs.composite
def add_dummy_layer_prefix_to(
    draw: hs.DrawFn, strat: hs.SearchStrategy[OptimizerTestData]
) -> OptimizerTestData:
    test_data = draw(strat)
    dummy_layer = '"batchnorm"'
    updated_tokenstream = dummy_layer + test_data.tokenstream
    return attrs.evolve(test_data, tokenstream=updated_tokenstream)
