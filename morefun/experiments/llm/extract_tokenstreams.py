import itertools
import sys
from pathlib import Path

from loguru import logger

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


def main() -> None:
    persistent_dir = Path("/workspaces/morefun/persistent")

    output_dir = persistent_dir / "tokenstreams"

    logger.remove()
    logger.add(sink=sys.stderr, level="INFO")

    num_runs = 5
    num_generations = 51

    for run_nr, gen_nr in itertools.product(range(num_runs), range(num_generations)):
        run_dir = persistent_dir / f"v2/run_{run_nr}"
        settings_path = run_dir / "settings.yaml"
        generation_checkpoint_path = (
            run_dir / "output" / f"{gen_nr}.generation_checkpoint"
        )

        if not generation_checkpoint_path.exists():
            print(f"File not found, skipping. Path={generation_checkpoint_path}")
            continue

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
    main()
