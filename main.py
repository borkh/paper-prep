import shutil
from pathlib import Path

from tqdm import tqdm

from model_search import ModelPlotter
from latex import LatexTemplate
from optuna_plotting import OptunaPlotter


class PaperPrep(ModelPlotter, LatexTemplate, OptunaPlotter):
    def __init__(
        self,
        start_path: Path,
        split: str,
        custom_metrics: dict = {},
        priority_min: list = [],
        priority_max: list = [],
        priority_order: str = "maximize",
        language: str = "english",
        significant_digits: int = 3,
    ):
        ModelPlotter.__init__(
            self,
            start_path,
            split,
            custom_metrics,
            priority_min,
            priority_max,
            priority_order,
        )
        LatexTemplate.__init__(
            self,
            language,
            significant_digits=significant_digits,
        )
        try:
            OptunaPlotter.__init__(self, start_path)
        except FileNotFoundError:
            print("No optuna database found.")
            self.study_names = []

        self.create_metric_plots()

    def generate_dir(self):
        """Create directory for paper and copy best models into it."""
        self.paper_path = self.start_path / "paper"
        self.paper_path.mkdir(exist_ok=True)

        pbar = tqdm(self.best_models)
        for model_data in pbar:
            rel_path = model_data["log_dir_relative"]
            pbar.set_description(f"Generating directory for .../{rel_path}")
            best_model_dir = model_data["log_dir"] / "best_model"

            # Create directory
            paper_model_dir = self.paper_path / "images" / rel_path
            paper_model_dir.mkdir(exist_ok=True, parents=True)

            # Copy files
            dir_content = [
                f
                for f in best_model_dir.iterdir()
                if not f.is_dir() and "tfevents" not in f.name
            ]
            for file in dir_content:
                shutil.copy(file, paper_model_dir)

    def generate_latex(
        self,
        metric_names={},
        hparams_names={},
        include_unspecified=True,
    ):
        """
        Generate LaTeX for each best model. The LaTeX is generated from a
        template and filled with the metrics, hyperparameters and images of
        the best model.
        One file is written to each best model directory and one file with
        all best models is written to the paper directory.
        """
        all_best_models_latex = ""

        pbar = tqdm(self.best_models)
        for model_data in pbar:
            pbar.set_description(
                f"Generating LaTeX for .../{model_data['log_dir_relative']}"
            )
            best_model_dir = model_data["log_dir"] / "best_model"
            metrics = model_data["metrics"]
            hparams = model_data["hparams"]

            image_files = [
                best_model_dir.parent / f.name
                for f in best_model_dir.glob("*.pdf")
            ]
            image_files = sorted(image_files, key=lambda x: x.name)
            image_files = [f.relative_to(self.start_path) for f in image_files]
            # remove "best_model" directory from path

            # Generate LaTeX
            latex_output = self.fill_latex_template(
                metrics,
                hparams,
                image_files,
                metric_names,
                hparams_names,
                include_unspecified,
                section_name=model_data["log_dir_relative"],
            )

            # Add LaTeX to all best models LaTeX
            all_best_models_latex += latex_output

            # Write LaTeX to file in best model directory
            with open(best_model_dir / "latex.tex", "w") as f:
                f.write(latex_output)

        # Write LaTeX to file in paper directory
        latex_path = self.start_path / "paper" / "chapter" / "ch_results.tex"
        latex_path.parent.mkdir(exist_ok=True, parents=True)
        with open(latex_path, "w") as f:
            f.write(all_best_models_latex)

    def generate_optuna_plots(self, hparam_names={}):
        """Generate plots for each study."""

        # Check if there are any studies.
        if not self.study_names:
            print("No optuna studies found. Unable to generate plots.")
            # return

        pbar = tqdm(self.study_names)
        for study_name in pbar:
            # check if study_name is in best_models
            model_data = [
                model_data
                for model_data in self.best_models
                if study_name in str(model_data["log_dir_relative"])
            ]
            if not model_data:
                continue
            model_data = model_data[0]
            pbar.set_description(f"Generating plots for study {study_name}")

            self.plot_param_importance(
                study_name,
                hparam_names,
                save_path=model_data["log_dir"]
                / "best_model"
                / "param_importance.pdf",
            )

            self.plot_optimization_history(
                study_name,
                save_path=model_data["log_dir"]
                / "best_model"
                / "optimization_history.pdf",
            )


if __name__ == "__main__":
    """Example usage. Replace artifacts_dir with your own directory."""

    artifacts_dir = "/path/to/tensorboard-model-logdir"
    #    custom_metrics = {"test/loss": "min"}

    pp = PaperPrep(
        artifacts_dir,
        split="test",
        # custom_metrics=custom_metrics,
        language="english",
        significant_digits=4,
    )
    metric_names = {
        "test_loss_epoch": "Testfehler",
        "test_accuracy_epoch": "Testgenauigkeit",
        "test_loss": "Testfehler",
        "test_accuracy": "Testgenauigkeit",
        "test_perplexity": "Testperplexität",
        "test_topk_accuracy": "Testgenauigkeit (Top-20)",
    }
    hparam_names = {
        "learning_rate": "Lernrate",
        "batch_size": "Batchgröße",
        "chunk_len": "Chunklänge",
        "stride": "Stride",
        "n_layers": "Anzahl Layer",
        "hidden_size": "Dim. verborgene Schicht",
        "fasttext_model_path": "fastText-Modell",
    }

    pp.generate_optuna_plots(hparam_names=hparam_names)
    pp.generate_dir()
    pp.generate_latex(
        metric_names=metric_names,
        hparams_names=hparam_names,
        include_unspecified=False,
    )
