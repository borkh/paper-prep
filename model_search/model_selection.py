import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

from .model_evaluation import ModelEvaluation


def generic_unknown_tag_handler(loader, tag_suffix, node):
    # Prepare a string representation of the node value
    if isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node)
    elif isinstance(node, yaml.MappingNode):
        value = loader.construct_mapping(node)
    else:
        value = None  # A default for unrecognized node types
    # Return a string that includes the tag and the value
    return f"!!{tag_suffix} {value}"


# Register the generic handler for all unknown tags
yaml.add_multi_constructor(
    "", generic_unknown_tag_handler, Loader=yaml.SafeLoader
)


class ModelSelection(ModelEvaluation):
    def __init__(
        self,
        start_path: Path,
        split: str,
        custom_metrics: dict = {},
        priority_min: list = [],
        priority_max: list = [],
        priority_order: str = "maximize",
    ):
        super().__init__(
            start_path,
            split,
            custom_metrics,
            priority_min,
            priority_max,
            priority_order,
        )

        self.best_models = self._select_best_models()

    def _select_best_models(self):
        best_models = []
        for i in self.data:
            log_dir = i["log_dir"]
            sorted_metrics_df = i["sorted_metrics_df"]

            if sorted_metrics_df.empty:
                continue
            best_model_dict = sorted_metrics_df.iloc[0].to_dict()

            # get hyperparameters from yaml file
            yaml_file = log_dir / best_model_dict["version"] / "hparams.yaml"
            if yaml_file.exists():
                with open(yaml_file) as f:
                    hparams = yaml.safe_load(f)
            else:
                hparams = {"hparams": {}}

            best_models.append(
                {
                    "log_dir": log_dir,
                    "log_dir_relative": log_dir.relative_to(self.start_path),
                    "metrics": best_model_dict,
                    "hparams": hparams["hparams"]
                    if "hparams" in hparams
                    else hparams,
                }
            )

        return best_models

    def remove_unused_checkpoints(self, no_confirm=False):
        """Removes the checkpoints of all models other than the best model.
        Might be useful to save space."""
        for model_data in self.best_models:
            if not no_confirm:
                confirm = input(
                    f"Are you sure you want to remove all checkpoints of .../{model_data['log_dir_relative']} except the best model? [y/n] "  # noqa
                )
                if confirm != "y":
                    continue

            log_dir = model_data["log_dir"]
            version_dirs = self._get_version_dirs(log_dir)

            # remove best model from list
            version_dirs.remove(log_dir / model_data["metrics"]["version"])
            version_dirs.sort()
            for version_dir in version_dirs:
                checkpoints = version_dir.glob("*checkpoint*")
                for checkpoint in checkpoints:
                    if checkpoint.is_dir():
                        print(f"Removing {checkpoint} ...")
                        shutil.rmtree(checkpoint)

    def link_best_models(self):
        pbar = tqdm(self.best_models)
        for data in pbar:
            pbar.set_description(
                f"Linking best model in .../{data['log_dir_relative']}"
            )
            # create symlink to best model
            path = data["log_dir"]
            metrics = data["metrics"]

            source_dir = path / metrics["version"]
            symlink_path = path / "best_model"

            if symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(source_dir, target_is_directory=True)

            # save metrics to yaml file in symlink directory
            with open(symlink_path / f"{self.split}_metrics.yaml", "w") as f:
                yaml.dump(metrics, f)
