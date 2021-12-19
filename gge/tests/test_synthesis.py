from pathlib import Path

import lark
import pytest

import gge.synthesis as syn

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def parser() -> lark.Lark:
    return lark.Lark.open(
        grammar_filename=str(DATA_DIR / "mesagrammar.lark"),
        parser="lalr",
    )


@pytest.fixture
def synthetizer() -> syn.BackboneSynthetizer:
    return syn.BackboneSynthetizer()


def test_conv2d(parser: lark.Lark, synthetizer: syn.BackboneSynthetizer) -> None:
    tokenstream = """
    conv2d filter_count 1 kernel_size 2 stride 3
    """

    tree = parser.parse(tokenstream)
    actual = synthetizer.transform(tree)

    expected = (syn.Conv2DLayer(filter_count=1, kernel_size=2, stride=3),)
    assert expected == actual


def test_dense(parser: lark.Lark, synthetizer: syn.BackboneSynthetizer) -> None:
    tokenstream = """
    dense 5
    """

    tree = parser.parse(tokenstream)
    actual = synthetizer.transform(tree)

    expected = (syn.DenseLayer(5),)
    assert expected == actual


def test_dropout(parser: lark.Lark, synthetizer: syn.BackboneSynthetizer) -> None:
    tokenstream = """
    dropout 0.7
    """

    tree = parser.parse(tokenstream)
    actual = synthetizer.transform(tree)

    expected = (syn.DropoutLayer(0.7),)
    assert expected == actual


def test_backbone(parser: lark.Lark, synthetizer: syn.BackboneSynthetizer) -> None:
    tokenstream = """
    conv2d filter_count 1 kernel_size 2 stride 3
    conv2d filter_count 4 kernel_size 5 stride 6
    dense 7
    dropout 0.8
    dense 9
    dropout 0.10
    """

    tree = parser.parse(tokenstream)
    actual = synthetizer.transform(tree)

    expected = (
        syn.Conv2DLayer(1, 2, 3),
        syn.Conv2DLayer(4, 5, 6),
        syn.DenseLayer(7),
        syn.DropoutLayer(0.8),
        syn.DenseLayer(9),
        syn.DropoutLayer(0.10),
    )

    assert expected == actual
