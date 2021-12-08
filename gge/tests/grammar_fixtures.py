from pathlib import Path

import pytest

from gge.grammars import Grammar

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def raw_metagrammar() -> str:
    metagrammar_path = DATA_DIR / "metagrammar.lark"
    return metagrammar_path.read_text()


@pytest.fixture
def sample_grammar(raw_grammar: str, raw_metagrammar: str) -> Grammar:
    return Grammar(
        raw_grammar=raw_grammar,
        raw_metagrammar=raw_metagrammar,
    )
