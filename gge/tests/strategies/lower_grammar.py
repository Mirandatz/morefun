import hypothesis.strategies as hs

import gge.layers as gl
from gge.name_generator import NameGenerator
from gge.tests.strategies import data_structures as tds

LowerGrammarParsingTestData = tds.ParsingTestData[tuple[gl.Layer, ...]]


@hs.composite
def random_flips(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.RandomFlip)

    possible_modes = [e.value for e in gl.FlipMode]

    mode = draw(hs.sampled_from(possible_modes))
    layer = gl.RandomFlip(name, gl.FlipMode(mode))
    tokenstream = f'"random_flip" "{mode}"'
    return LowerGrammarParsingTestData(tokenstream, (layer,))


@hs.composite
def random_rotations(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.RandomRotation)
    factor = draw(hs.floats(min_value=0, max_value=1))
    layer = gl.RandomRotation(name, factor)
    tokenstream = f'"random_rotation" "{factor}"'
    return LowerGrammarParsingTestData(tokenstream, (layer,))


@hs.composite
def resizings(
    draw: hs.DrawFn, *, name_gen: NameGenerator | None = None
) -> LowerGrammarParsingTestData:
    name_gen = name_gen or NameGenerator()
    name = name_gen.gen_name(gl.Resizing)
    target_height = draw(hs.integers(min_value=1, max_value=9999))
    target_width = draw(hs.integers(min_value=1, max_value=9999))
    tokenstream = f'"resizing" "height" "{target_height}" "width" "{target_width}"'
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
    tokenstream = f'"random_crop" "height" "{height}" "width"  "{width}"'
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
    tokenstream = f'"conv" "filter_count" "{filter_count}" "kernel_size" "{kernel_size}" "stride" "{stride}"'
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
    tokenstream = f'"maxpool" "pool_size" "{pool_size}" "stride" "{stride}"'
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
    tokenstream = f'"avgpool" "pool_size" "{pool_size}" "stride" "{stride}"'
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
