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
def synthetizer() -> syn.Synthetizer:
    return syn.Synthetizer()


def test_conv2d(parser: lark.Lark, synthetizer: syn.Synthetizer) -> None:
    tokenstream = """
    conv2d filter_count 1 kernel_size 2 stride 3
    """

    tree = parser.parse(tokenstream)
    actual = synthetizer.transform(tree)

    expected = (syn.Conv2dNode(filter_count=1, kernel_size=2, stride=3),)
    assert expected == actual


def test_dense(parser: lark.Lark, synthetizer: syn.Synthetizer) -> None:
    tokenstream = """
    dense 5
    """

    tree = parser.parse(tokenstream)
    actual = synthetizer.transform(tree)

    expected = (syn.DenseNode(5),)
    assert expected == actual


def test_dropout(parser: lark.Lark, synthetizer: syn.Synthetizer) -> None:
    tokenstream = """
    dropout 0.7
    """

    tree = parser.parse(tokenstream)
    actual = synthetizer.transform(tree)

    expected = (syn.DropoutNode(0.7),)
    assert expected == actual


def test_backbone(parser: lark.Lark, synthetizer: syn.Synthetizer) -> None:
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
        syn.Conv2dNode(1, 2, 3),
        syn.Conv2dNode(4, 5, 6),
        syn.DenseNode(7),
        syn.DropoutNode(0.8),
        syn.DenseNode(9),
        syn.DropoutNode(0.10),
    )

    assert expected == actual
