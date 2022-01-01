import gge.layers as gl
import gge.name_generator as ng


def test_first_prefix() -> None:
    gen = ng.NameGenerator()
    first = gen.create_name("k")
    assert "k_0" == first

    second = gen.create_name("k")
    assert "k_1" == second


def test_multiple_prefixes() -> None:
    gen = ng.NameGenerator()
    _ = gen.create_name("x")

    actual = gen.create_name("y")
    assert "y_0" == actual


def test_type() -> None:
    gen = ng.NameGenerator()
    actual_first = gen.create_name(gl.Conv2D)
    expected_first = "Conv2D_0"
    assert expected_first == actual_first

    actual_second = gen.create_name(gl.Conv2D)
    expected_second = "Conv2D_1"
    assert expected_second == actual_second
