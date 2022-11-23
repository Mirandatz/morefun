# import pathlib

# from loguru import logger

# import gge.evolutionary.fitnesses as gf
# import gge.experiments.settings as gset
# import gge.persistence
# import gge.phenotypes


# def process_run_directory(run_directory: pathlib.Path) -> None:
#     settings_path = run_directory / "settings.yaml"
#     settings = gset.load_gge_settings(settings_path)
#     gset.configure_logger(settings.output)
#     gset.configure_tensorflow(settings.tensorflow)

#     logger.info(f"processing: {run_directory}")

#     output_path = run_directory / "output"
#     last_generation_output = gge.persistence.load_latest_generational_artifacts(
#         output_path
#     )

#     best = max(
#         last_generation_output.get_fittest().items(),
#         key=lambda genotype_fitness: genotype_fitness[1].to_effective_fitnesses_dict()[
#             "train_loss"
#         ],
#     )

#     genotype = best[0]

#     phenotype = gge.phenotypes.translate(genotype, settings.grammar)
#     model = gf.make_classification_model(
#         phenotype,
#         input_shape=settings.dataset.input_shape,
#         class_count=settings.dataset.class_count,
#     )
#     json = model.to_json()

#     filename = (output_path / genotype.unique_id.hex).with_suffix(".json")
#     filename.write_text(json)


# def main() -> None:
#     process_run_directory(
#         pathlib.Path("/workspaces/gge/gge/playground/gitignored/results/cifar10/seed_0")
#     )


# if __name__ == "__main__":
#     main()
