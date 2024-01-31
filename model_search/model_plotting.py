from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from .model_selection import ModelSelection
from tqdm import tqdm

sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")


class ModelPlotter(ModelSelection):
    def __init__(
        self,
        start_path: Path,
        split: str,
        custom_metrics: dict = {},
        priority_min=[],
        priority_max=[],
        priority_order="maximize",
    ):
        super().__init__(
            start_path,
            split,
            custom_metrics,
            priority_min,
            priority_max,
            priority_order,
        )
        self.link_best_models()

    def _plot_train_val_curve(
        self, train_metrics_df, val_metrics_df, save_dir
    ):
        """Plot the training and validation curves for each metric."""

        for train_metric in train_metrics_df.columns:
            metric = train_metric.split("_step")[0]
            metric = metric.split("train")[-1]
            # remove any leading underscores or slashes
            metric = metric.lstrip("_").lstrip("/")

            # only continue if the metric is only once in the train_metric
            # string (e.g. train/train_loss is used for epochs which is not
            # wanted here)
            if train_metric.count("train") > 1:
                continue

            # Find a matching validation metric in the validation DataFrame
            # columns
            matched_val_metrics = [
                val_metric
                for val_metric in val_metrics_df.columns
                if metric in val_metric
            ]

            # There might be multiple matching validation metrics; choose the
            # first match for simplicity
            if matched_val_metrics:
                val_metric = matched_val_metrics[
                    0
                ]  # Use the first matching validation metric

                plt.figure(figsize=(5, 3))
                plt.plot(
                    train_metrics_df.index,
                    train_metrics_df[train_metric],
                    label=f"Train {metric.capitalize()}",
                )
                plt.plot(
                    val_metrics_df.index,
                    val_metrics_df[val_metric],
                    label=f"Val {metric.capitalize()}",
                )
                plt.xlabel("Step")
                plt.ylabel(metric.capitalize())
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                # Save the plot
                plt.savefig(save_dir / f"{metric}.pdf")
                plt.close()

    def create_metric_plots(self):
        pbar = tqdm(self.best_models)
        for data in pbar:
            pbar.set_description(
                f"Creating metric plots for best model in .../{data['log_dir_relative']}"  # noqa
            )
            path = data["log_dir"]
            best_model_path = path / "best_model"
            tf_events_files = self._get_tf_events_files(best_model_path)
            for tfevent in tf_events_files:
                train_df = self.extract_data_from_event_file(
                    str(tfevent),
                    split="train",
                    exclude="epoch",
                )
                train_metrics_df = self._extract_metrics_from_dataframe(
                    train_df
                )

                val_df = self.extract_data_from_event_file(
                    str(tfevent),
                    split="val",
                    exclude="epoch",
                )
                val_metrics_df = self._extract_metrics_from_dataframe(val_df)

                self._plot_train_val_curve(
                    train_metrics_df,
                    val_metrics_df,
                    save_dir=best_model_path,
                )
