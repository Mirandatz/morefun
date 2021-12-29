import gge.name_generator as gge_namegen


def test_first_prefix() -> None:
    gen = gge_namegen.NameGenerator()
    first = gen.create_name("k")
    assert "k_0" == first

    second = gen.create_name("k")
    assert "k_1" == second


def test_multiple_prefixes() -> None:
    gen = gge_namegen.NameGenerator()
    _ = gen.create_name("x")

    actual = gen.create_name("y")
    assert "y_0" == actual
