"""
This module assumes that fitnesses must be MINIMIZED.
"""

import pickle
from pathlib import Path
from typing import Annotated

import typer

from morefun.evolutionary.fitnesses import TrainLoss


def main(
    train_loss_instance_path: Annotated[
        Path, typer.Option(help="Path to the train loss instance pickle file")
    ],
    phenotype_path: Annotated[
        Path, typer.Option(help="Path to the phenotype pickle file")
    ],
    eval_result_path: Annotated[Path, typer.Option(help="Path to eval result file")],
) -> int:
    train_loss_instance = pickle.loads(train_loss_instance_path.read_bytes())
    assert isinstance(train_loss_instance, TrainLoss)

    phenotype = pickle.loads(phenotype_path.read_bytes())

    result = train_loss_instance._evaluate(phenotype)

    eval_result_path.parent.mkdir(parents=True, exist_ok=True)
    eval_result_path.write_bytes(pickle.dumps(result))

    return 0


if __name__ == "__main__":
    typer.run(main)
