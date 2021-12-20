from pathlib import Path

import lark
import pytest

import gge.synthesis as syn

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def parser() -> lark.Lark:
    return lark.Lark(
        grammar=(DATA_DIR / "mesagrammar.lark").read_text(),
        parser="lalr",
        maybe_placeholders=True,
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


def test_merge_(parser: lark.Lark, synthetizer: syn.BackboneSynthetizer) -> None:
    tokenstream = """
    merge dense 2
    """

    tree = parser.parse(tokenstream)
    actual = synthetizer.transform(tree)

    expected = (syn.Merge(), syn.DenseLayer(2))
    assert expected == actual


def test_fork(parser: lark.Lark, synthetizer: syn.BackboneSynthetizer) -> None:
    tokenstream = """
    dense 1 fork
    """

    tree = parser.parse(tokenstream)
    actual = synthetizer.transform(tree)

    expected = (syn.DenseLayer(1), syn.Fork())
    assert expected == actual


def test_merge_and_fork(
    parser: lark.Lark, synthetizer: syn.BackboneSynthetizer
) -> None:
    tokenstream = """
    merge dense 5 fork
    """

    tree = parser.parse(tokenstream)
    actual = synthetizer.transform(tree)

    expected = (syn.Merge(), syn.DenseLayer(5), syn.Fork())
    assert expected == actual


def test_simple_backbone(
    parser: lark.Lark, synthetizer: syn.BackboneSynthetizer
) -> None:
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


def test_complex_backbone(
    parser: lark.Lark, synthetizer: syn.BackboneSynthetizer
) -> None:
    tokenstream = """
    conv2d filter_count 1 kernel_size 2 stride 3 fork
    merge conv2d filter_count 4 kernel_size 5 stride 6 fork
    merge conv2d filter_count 7 kernel_size 8 stride 9

    dense 10
    dropout 0.11 fork

    dense 12
    dropout 0.13 fork

    merge dense 14
    dropout 0.15

    """

    tree = parser.parse(tokenstream)
    actual = synthetizer.transform(tree)

    a = [syn.Conv2DLayer(1, 2, 3), syn.Fork()]
    b = [syn.Merge(), syn.Conv2DLayer(4, 5, 6), syn.Fork()]
    c = [syn.Merge(), syn.Conv2DLayer(7, 8, 9)]
    d = [syn.DenseLayer(10), syn.DropoutLayer(0.11), syn.Fork()]
    e = [syn.DenseLayer(12), syn.DropoutLayer(0.13), syn.Fork()]
    f = [syn.Merge(), syn.DenseLayer(14), syn.DropoutLayer(0.15)]

    expected = tuple([*a, *b, *c, *d, *e, *f])

    assert expected == actual
