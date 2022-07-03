import pathlib
import pickle

import keras.api._v2.keras as keras  # noqa
import typer

import gge.grammars as gr
import gge.layers as gl
import gge.phenotypes as phenos


def main(
    genotypes_dir: pathlib.Path = typer.Option(
        ...,
        "--genotypes-dir",
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        writable=True,
    ),
    grammar_path: pathlib.Path = typer.Option(
        ...,
        "--grammar-path",
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
    ),
) -> None:
    assert genotypes_dir.is_dir()
    assert grammar_path.is_file()

    grammar = gr.Grammar(grammar_path.read_text())

    for genotype_path in genotypes_dir.glob("*.genotype"):
        genotype = pickle.loads(genotype_path.read_bytes())
        phenotype = phenos.translate(genotype, grammar)
        in_tensor, out_tensor = phenos.make_input_output_tensors(
            phenotype,
            gl.make_input(32, 32, 3),
        )
        model = keras.Model(in_tensor, out_tensor)
        keras.utils.plot_model(
            model,
            to_file=genotype_path.with_suffix(".png"),
            show_shapes=True,
            show_layer_names=False,
        )


def run_from_script() -> None:
    base_dir = pathlib.Path(__file__).parent / "cifar10"
    pop_dir = base_dir / "initial_population"
    grammar_path = base_dir / "grammar.lark"
    main(pop_dir, grammar_path)


if __name__ == "__main__":
    run_from_script()


# if we want to use this from the cli:
# if __name__ == "__main__":
#     typer.run(main)
