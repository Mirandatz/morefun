from pathlib import Path

import lark

import pytest

from gge import synthetizer

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def synthesis_parser() -> lark.Lark:
    return lark.Lark.open(DATA_DIR / "synthetizer.lark")


def test_parse_sample(synthesis_parser: lark.Lark):
    phenotype = (
        '"conv2d" "filter_count" (5) "kernel_size" (2) "stride" (1)'
        '"dropout" (0.5)'
        '"conv2d" "filter_count" (2) "kernel_size" (5) "stride" (2)'
    )
    tree = synthesis_parser.parse(phenotype)
    synth = synthetizer.Synthetizer()
    actual = synth.transform(tree)

    expected = (
        synthetizer.ConvLayer(
            synthetizer.FilterCount(5), synthetizer.KernelSize(2), synthetizer.Stride(1)
        ),
        synthetizer.DropoutLayer(synthetizer.Probability(0.5)),
        synthetizer.ConvLayer(
            synthetizer.FilterCount(2), synthetizer.KernelSize(5), synthetizer.Stride(2)
        ),
    )

    assert actual == expected
