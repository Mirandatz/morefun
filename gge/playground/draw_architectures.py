import pathlib

import tensorflow as tf
from keras import Model as KerasModel
from loguru import logger

import gge.evolutionary.fitnesses as gf
import gge.evolutionary.generations
import gge.experiments.settings as gset
import gge.paths
import gge.phenotypes
import gge.randomness
import gge.redirection


def get_gitignored_dir() -> pathlib.Path:
    root = gge.paths.get_project_root_dir()
    return root / "gge" / "playground" / "gitignored"


def main() -> None:
    for run_id in ["fm_0", "fm_1", "fm_2", "fm_3", "fm_4"]:
        run_dir = get_gitignored_dir() / "cifar10" / run_id
        output_dir = run_dir / "output"
        settings_path = run_dir / "settings.yaml"

        settings = gset.load_gge_settings(settings_path)
        gset.configure_logger(settings.output)
        gset.configure_tensorflow(settings.tensorflow)

        last_checkpoint = gge.evolutionary.generations.GenerationCheckpoint.load(
            gge.paths.get_generation_checkpoint_path(
                output_dir=output_dir,
                generation_number=51,
            )
        )
        for individual in last_checkpoint.get_population():
            if individual.genotype.unique_id.hex != "7a8c3b21152b4b30b1d62bf48ca77d5e":
                continue

            logger.info(f"processing genotype=<{individual.genotype.unique_id.hex}>")

            model: KerasModel = gf.make_classification_model(
                phenotype=individual.phenotype,
                input_shape=settings.dataset.input_shape,
                class_count=settings.dataset.class_count,
            )

            plot_path = gge.paths.get_architecture_plot_path(
                output_dir=output_dir,
                uuid=individual.genotype.unique_id,
            )

            # dot = tf.keras.utils.model_to_dot(model)
            # print(dot)

            tf.keras.utils.plot_model(
                model,
                to_file=plot_path,
                show_layer_names=True,
                show_shapes=True,
                show_layer_activations=True,
            )


if __name__ == "__main__":
    main()
