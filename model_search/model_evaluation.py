from .model_analysis import ModelAnalysis
from pathlib import Path
from tqdm import tqdm


class ModelEvaluation(ModelAnalysis):
    def __init__(
        self,
        start_path: Path,
        split: str,
        custom_metrics: dict = {},
        priority_min: list = [],
        priority_max: list = [],
        priority_order: str = "maximize",
    ):
        super().__init__(start_path, split)
        self.custom_metrics = custom_metrics
        self.priority_min = priority_min
        self.priority_max = priority_max
        self.priority_order = priority_order

        self.min_metrics = [
            "loss",
            "mae",
            "mape",
            "mse",
            "rmse",
            "perplexity",
        ]
        self.max_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "topk",
            "top_k",
        ]

        # Add any custom metrics to the base metrics
        for metric in self.custom_metrics:
            if self.custom_metrics[metric] == "min":
                self.min_metrics.append(metric)
            elif self.custom_metrics[metric] == "max":
                self.max_metrics.append(metric)
            else:
                raise ValueError(
                    "Custom_metrics values must be either 'min' or 'max'."
                )

        self.data = self._create_sorted_metrics_dataframe()

    def _extract_metrics_from_dataframe(self, df):
        """Extract metrics from dataframe and add them to the dataframe."""

        # first check if the dataframe is not empty
        if df.empty:
            return df

        # drop any columns where a metric is not in the name of the column
        # except the version column
        all_metrics = (
            self.min_metrics
            + self.max_metrics
            + self.priority_min
            + self.priority_max
            + list(self.custom_metrics.keys())
        )
        df = df.drop(
            columns=[
                col
                for col in df.columns
                if not any(metric in col for metric in all_metrics)
                and col != "version"
            ]
        )

        return df

    def _sort_metrics_dataframe(
        self,
        df,
    ):
        df = self._extract_metrics_from_dataframe(df)

        for prio in self.priority_min:
            if prio not in self.min_metrics:
                self.min_metrics.append(prio)
        for prio in self.priority_max:
            if prio not in self.max_metrics:
                self.max_metrics.append(prio)

        # Initialize empty lists to hold column names for sorting
        sort_cols_min = []
        sort_cols_max = []

        # Classify each column based on whether it should be minimized or
        # maximized
        for col in df.columns:
            if any(min_metric in col for min_metric in self.min_metrics):
                sort_cols_min.append(col)
            elif any(max_metric in col for max_metric in self.max_metrics):
                sort_cols_max.append(col)

        if sort_cols_min == [] and sort_cols_max == []:
            pass
            # print(
            #     "No metric in the dataframe matches the min_metrics or "
            #     "max_metrics lists.\nYou may need to add a custom metric to "
            #     "the custom_metrics dictionary or provide it in the "
            #     "\npriority_min or priority_max lists. "
            #     "The log directory may also be empty."
            # )

        # Add any priority metrics to the beginning of the sort lists
        for priority_metric in self.priority_min:
            # if the priority metric is in the name of a column, remove it from
            # the list and add it to the beginning of the list
            for col in sort_cols_min:
                if priority_metric in col:
                    sort_cols_min.remove(col)
                    sort_cols_min.insert(0, col)

        for priority_metric in self.priority_max:
            # if the priority metric is in the name of a column, remove it from
            # the list and add it to the beginning of the list
            for col in sort_cols_max:
                if priority_metric in col:
                    sort_cols_max.remove(col)
                    sort_cols_max.insert(0, col)

        # Adjust sorting logic based on priority_order
        if self.priority_order == "minimize":
            sorted_metrics_df = df.sort_values(
                by=sort_cols_min + sort_cols_max,
                ascending=[True] * len(sort_cols_min)
                + [False] * len(sort_cols_max),
                ignore_index=True,
            )
        elif self.priority_order == "maximize":
            sorted_metrics_df = df.sort_values(
                by=sort_cols_max + sort_cols_min,
                ascending=[False] * len(sort_cols_max)
                + [True] * len(sort_cols_min),
                ignore_index=True,
            )
        else:
            raise ValueError(
                "Priority_order must be either 'minimize' or 'maximize'."
            )

        return sorted_metrics_df

    def _create_sorted_metrics_dataframe(self):
        new_data = []
        pbar = tqdm(self.data)
        for data in pbar:
            pbar.set_description(
                f"Sorting dataframes .../{data['log_dir'].relative_to(self.start_path)}"  # noqa
            )
            sorted_metrics_df = self._sort_metrics_dataframe(data["dataframe"])
            new_data.append(
                {
                    "log_dir": data["log_dir"],
                    "dataframe": data["dataframe"],
                    "sorted_metrics_df": sorted_metrics_df,
                }
            )

        return new_data
