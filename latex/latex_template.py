import math
from pathlib import Path


class LatexTemplate:
    """
    A class to generate LaTeX code for reports, including figures and tables,
    with support for different languages.
    """

    def __init__(self, language="english", significant_digits=3):
        self.language = language
        self.significant_digits = significant_digits
        self.translations = self._get_translations()
        self._use_comma = True if language == "german" else False

    @staticmethod
    def _get_translations():
        """
        Returns a dictionary of translations for supported languages.
        """
        return {
            "english": {
                "metric": "Metric",
                "value": "Value",
                "hyperparameter": "Hyperparameter",
                "optuna": "Optimization history and parameter importance.",
                "caption": "Training progress, metrics on the test dataset and hyperparameters for the best model...",  # noqa
            },
            "german": {
                "metric": "Metrik",
                "value": "Wert",
                "hyperparameter": "Hyperparameter",
                "optuna": "Optimierungshistorie und Parameterwichtigkeit.",
                "caption": "Trainingsverlauf, Metriken auf dem Testdatensatz und Hyperparameter für das beste Modell...",  # noqa
            },
        }

    def _translate(self, keyword):
        """
        Translates a keyword to the specified language.
        """
        return self.translations[self.language].get(keyword, keyword)

    def _rename_and_filter(
        self, original_dict, name_dict, include_unspecified
    ):
        """
        Rename keys in the original dictionary based on the provided name_dict.
        If include_unspecified is False, only include keys that are present in
        name_dict.
        """
        new_dict = {}
        for k, v in original_dict.items():
            if k in name_dict or include_unspecified:
                new_key = name_dict.get(k, k).replace("_", "\\_")
                new_dict[new_key] = v
        return new_dict

    def _generate_minipage_for_image(self, image_file, index):
        """
        Generates a minipage for a single image file.
        """
        return (
            f"\t\\begin{{minipage}}[c]{{0.48\\textwidth}}\n"
            f"\t\t\\centering\n"
            f"\t\t\\includegraphics[width=\\textwidth]{{images/{image_file}}}\n"  # noqa
            f"\t\\end{{minipage}}" + ("\n\t\\hfill" if index % 2 == 0 else "")
        )

    def _generate_image_minipages(self, image_files):
        """
        Generates LaTeX code for including a list of images in minipages.
        """
        return "\n".join(
            [
                self._generate_minipage_for_image(image_file, index)
                for index, image_file in enumerate(image_files, start=1)
            ]
        )

    def _generate_image_figure(self, image_files, caption, label=""):
        """
        Generates a figure environment for a set of images.
        """
        image_minipages = self._generate_image_minipages(image_files)
        label = f"fig:{label}" if label else ""
        return (
            f"\\begin{{figure}}[h!]\n"
            f"\t\\centering\n"
            f"{image_minipages}\n"
            f"\t\\caption{{{caption}}}\n"
            f"\t\\label{{{label}}}\n"
            f"\\end{{figure}}"
        )

    def _generate_table_rows(self, data):
        """
        Generates table rows from a dictionary of data.
        """
        return "\n".join(
            f"\t\t\t{key} & {value} \\\\" for key, value in data.items()
        )

    def _generate_table_figure(self, data, title):
        """
        Generates a table figure from data.
        """
        rows = self._generate_table_rows(data)
        return (
            f"\t\\begin{{minipage}}[c]{{0.48\\textwidth}}\n"
            f"\t\t\\centering\n"
            f"\t\t\\usefont{{T1}}{{pfr}}{{l}}{{n}}\\selectfont\n"
            f"\t\t\\begin{{tabular}}{{@{{}}ll@{{}}}}\n"
            f"\t\t\t\\toprule\n"
            f"\t\t\t\\textbf{{{title}}} & \\textbf{{{self._translate('value')}}} \\\\\n"  # noqa
            f"\t\t\t\\midrule\n"
            f"{rows}\n"
            f"\t\t\t\\bottomrule\n"
            f"\t\t\\end{{tabular}}\n"
            f"\t\\end{{minipage}}"
        )

    def _round_to_significant_digits_latex(self, num):
        if num == 0:
            return "$0$"

        # Apply scientific notation for very small or very large numbers
        if abs(num) < 1e-3 or abs(num) > 1e6:
            exponent = int(f"{num:e}".split("e")[-1])
            # Round the base to the specified number of significant digits
            base = round(num / (10**exponent), self.significant_digits - 1)
            formatted_base = f"{{:.{self.significant_digits - 1}f}}".format(
                base
            )

            if self._use_comma:
                formatted_base = formatted_base.replace(".", ",")

            return f"${formatted_base} \\times 10^{{{exponent}}}$"
        else:
            if abs(num) >= 1:
                rounded_num = round(
                    num,
                    self.significant_digits
                    - int(math.floor(math.log10(abs(num))))
                    - 1,
                )
                formatted_num = "{:0." + str(self.significant_digits) + "f}"
                formatted_num = (
                    formatted_num.format(rounded_num).rstrip("0").rstrip(".")
                )
            else:
                formatted_num = "{:0." + str(self.significant_digits) + "f}"
                formatted_num = formatted_num.format(num)

            if self._use_comma:
                formatted_num = formatted_num.replace(".", ",")

            return f"${formatted_num}$"

    def _convert_paths_to_stem(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, Path):
                stem = value.stem
            elif isinstance(value, str):
                try:
                    path = Path(value)
                    if (
                        path.exists() or path.is_absolute()
                    ):  # Check if it's a valid path
                        stem = path.stem
                    else:
                        continue  # If it's not a valid path, skip to next item
                except Exception:
                    continue  # If it's not a valid path, skip to the next item
            else:
                continue  # If it's not a path, skip to the next item

            # Replace underscores with escaped underscores for LaTeX
            dictionary[key] = stem.replace("_", "\\_")

        return dictionary

    def fill_latex_template(
        self,
        metrics,
        hyperparameters,
        image_files,
        metric_names={},
        hparams_names={},
        include_unspecified=True,
        section_name=None,
    ):
        """
        Fills a LaTeX template with metrics, hyperparameters, and images.
        """
        esc_section_name = (
            str(section_name).replace("_", "\\_") if section_name else None
        )
        section = (
            f"\n\n\\section{{{esc_section_name}}}\n\n"
            if esc_section_name
            else ""
        )
        metrics = self._rename_and_filter(
            metrics, metric_names, include_unspecified
        )
        hyperparameters = self._rename_and_filter(
            hyperparameters, hparams_names, include_unspecified
        )
        hyperparameters = self._convert_paths_to_stem(hyperparameters)

        # Round metrics and hyperparameters
        metrics = {
            k: self._round_to_significant_digits_latex(v)
            if isinstance(v, float)
            else v
            for k, v in metrics.items()
        }

        hyperparameters = {
            k: self._round_to_significant_digits_latex(v)
            if isinstance(v, float)
            else v
            for k, v in hyperparameters.items()
        }

        # Sort metrics and hyperparameters
        metrics = dict(sorted(metrics.items()))
        hyperparameters = dict(sorted(hyperparameters.items()))

        # Categorize and process images
        metric_image_files = [
            image
            for image in image_files
            if any(image.stem in metric for metric in metric_names.keys())
        ]
        optuna_images = [
            image
            for image in image_files
            if "param_importance" in image.stem
            or "optimization_history" in image.stem
        ]
        other_images = [
            image
            for image in image_files
            if image not in optuna_images + metric_image_files
        ]

        # Create figures for images
        optuna_images_figure = (
            self._generate_image_figure(
                optuna_images,
                self._translate("optuna"),
                f"optuna_{section_name}",
            )
            if optuna_images
            else ""
        )
        metric_images_figure = self._generate_image_minipages(
            metric_image_files
        )
        additional_images_figure = (
            self._generate_image_figure(
                other_images, "[MISC]", f"misc_{section_name}"
            )
            if other_images
            else ""
        )

        # Create combined figure for metrics and hyperparameters tables
        metrics_table = self._generate_table_figure(
            metrics, self._translate("metric")
        )
        hyperparameters_table = self._generate_table_figure(
            hyperparameters, self._translate("hyperparameter")
        )
        tables_figure = (
            f"\\begin{{figure}}[h!]\n"
            f"\t\\centering\n"
            f"{metric_images_figure}\n"
            f"{metrics_table}\n"
            f"\t\\hfill\n"
            f"{hyperparameters_table}\n"
            f"\t\\caption{{{self._translate('caption')}}}\n"
            f"\t\\label{{fig:metrics_hparams_{section_name}}}\n"
            f"\\end{{figure}}\n"
        )

        # Combine all elements
        return (
            f"{section}"
            f"{optuna_images_figure}\n"
            f"{tables_figure}\n"
            f"{additional_images_figure}\n"
        )
