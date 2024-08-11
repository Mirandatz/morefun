#!/bin/env python

from pathlib import Path
from typing import Annotated

import typer

from morefun.composite_genotypes import CompositeGenotype
from morefun.evolutionary.generations import EvaluatedGenotype, GenerationCheckpoint
from morefun.grammars.structured_grammatical_evolution import (
    map_to_tokenstream,
)
from morefun.grammars.upper_grammars import Grammar as UpperGrammar
from morefun.neural_networks.connections import (
    ConnectionsSchema,
    MergeStrategy,
    ReshapeStrategy,
)


def load_genotypes(generation_checkpoint_path: Path) -> tuple[EvaluatedGenotype, ...]:
    return GenerationCheckpoint.load(generation_checkpoint_path).get_population()


def text_encode(text: str, tag: str) -> str:
    return f"<{tag}>{text}</{tag}>"


def convert_forks_mask_to_tokenstream(forks_mask: tuple[bool, ...]) -> str:
    return ",".join(str(b) for b in forks_mask)


def convert_merge_strategy_to_tokenstream(merge_strategy: MergeStrategy) -> str:
    return merge_strategy.name


def convert_reshape_strategy_to_tokenstream(reshape_strategy: ReshapeStrategy) -> str:
    return reshape_strategy.name


def convert_connection_schema_to_tokenstream(
    connection_schema: ConnectionsSchema,
) -> str:
    tokenstream = ""

    for mp in connection_schema.merge_params:
        tokenstream += text_encode(
            text=convert_forks_mask_to_tokenstream(mp.forks_mask),
            tag="forks_mask",
        )
        tokenstream += text_encode(
            text=convert_merge_strategy_to_tokenstream(mp.merge_strategy),
            tag="merge_strategy",
        )

    return tokenstream


def convert_composite_genotype_to_tokenstream(
    cg: CompositeGenotype,
    grammar: UpperGrammar,
) -> str:
    tokenstream = text_encode(
        text=map_to_tokenstream(cg.backbone_genotype, grammar),
        tag="backbone_genotype",
    )
    tokenstream += text_encode(
        text=convert_connection_schema_to_tokenstream(cg.connections_genotype),
        tag="connections_genotype",
    )
    return tokenstream


def load_grammar(settings_path: Path) -> UpperGrammar:
    import yaml

    with settings_path.open("r") as file:
        yaml_dict = yaml.safe_load(file)
    raw_grammar: str = yaml_dict["grammar"]

    return UpperGrammar(raw_grammar)


def main(
    generation_checkpoint_path: Annotated[
        Path,
        typer.Option(
            "-g",
            "--generation-checkpoint",
            file_okay=True,
            exists=True,
            readable=True,
            dir_okay=False,
        ),
    ],
    settings_path: Annotated[
        Path,
        typer.Option(
            "-s",
            "--settings-path",
            file_okay=True,
            exists=True,
            readable=True,
            dir_okay=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output-dir",
            file_okay=False,
            dir_okay=True,
        ),
    ],
) -> None:
    grammar = load_grammar(settings_path)
    genotypes = load_genotypes(generation_checkpoint_path)
    tokenstreams = {
        evaluated_genotype.genotype.unique_id: convert_composite_genotype_to_tokenstream(
            evaluated_genotype.genotype, grammar
        )
        for evaluated_genotype in genotypes
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for unique_id, tokenstream in tokenstreams.items():
        file_path = output_dir / unique_id.hex
        file_path.write_text(tokenstream)


if __name__ == "__main__":
    typer.run(main)
