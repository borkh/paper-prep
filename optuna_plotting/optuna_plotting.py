import matplotlib.pyplot as plt
import optuna
import seaborn as sns

sns.set_context("paper", font_scale=1.5)
sns.set(style="whitegrid")


class OptunaPlotter:
    def __init__(self, start_path):
        self.start_path = start_path
        self._search_database()
        self.storage = self._search_database()
        self.study_names = self._get_study_names()

    def _search_database(self):
        for path in self.start_path.rglob("optuna.db"):
            return f"sqlite:///{path}"
        if self.db_path is None:
            raise FileNotFoundError(
                f"No optuna database found in {self.start_path}."
            )

    def _get_study_names(self):
        study_summaries = optuna.get_all_study_summaries(storage=self.storage)
        study_names = [summary.study_name for summary in study_summaries]
        return study_names

    def _rename_and_replace(self, original_dict, name_dict):
        """Rename and filter dictionary keys based on provided dictionary.
        If include_unspecified is True, all keys that are not specified in the
        name_dict are included in the returned dictionary.
        """
        return {
            name_dict.get(k, k.replace("_", " ")): v
            for k, v in original_dict.items()
        }

    def plot_param_importance(
        self,
        study_name,
        hparam_names={},
        save_path=None,
    ):
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        param_importances = optuna.importance.get_param_importances(study)
        # remove underscore from parameter names
        param_importances = self._rename_and_replace(
            param_importances, hparam_names
        )

        plt.figure(figsize=(5, 3))
        plt.bar(param_importances.keys(), param_importances.values())
        plt.ylabel("Importance")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_optimization_history(
        self,
        study_name,
        save_path=None,
    ):
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        best_values = [
            t.value if t.value is not None else float("inf")
            for t in study.trials
        ]

        if "ec" in study_name:
            best_values_so_far = [
                max(best_values[: i + 1]) for i in range(len(best_values))
            ]
        else:
            best_values_so_far = [
                min(best_values[: i + 1]) for i in range(len(best_values))
            ]

        obj_val = (
            "Validation Accuracy" if "ec" in study_name else "Validation Loss"
        )  # noqa

        # Plot objective values as points
        plt.figure(figsize=(5, 3))
        plt.scatter(
            range(1, len(best_values) + 1),
            best_values,
            label=obj_val,
            alpha=0.6,
        )

        # Plot the best value as a line
        plt.plot(
            range(1, len(best_values_so_far) + 1),
            best_values_so_far,
            label="Best Value",
            color="red",
        )

        plt.xlabel("Trial")
        plt.ylabel(obj_val)
        plt.legend()
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_parallel_coordinate(self, study_name, params, save_path=None):
        # TODO: might need some improvements
        study = optuna.load_study(study_name=study_name, storage=self.storage)
        optuna.visualization.matplotlib.plot_parallel_coordinate(
            study, params=params
        )
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
