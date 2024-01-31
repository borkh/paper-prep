# paper-prep

## Overview
Paper-Prep is a suite of tools designed to automate the creation of plots, LaTeX tables, and various other elements needed in academic paper preparation. This project aims to streamline the process of preparing papers by providing easy-to-use scripts and functions. Currently only works for tensorboard logs (tfevent files).

### Important Notice:
This repository is currently a work in progress. As such, it has not been extensively tested across a wide range of scenarios. Therefore, users should be aware that it might not perform as expected in every unique setting or with different types of data and setups.

### Tools

- **model_search_base.py**: Features the `ModelSearchBase` class for organizing and retrieving model-related directories and data.
- **model_analysis.py**: Provides the `ModelAnalysis` class for analyzing model data, including functions for data extraction and dataframe creation.
- **model_evaluation.py**: Contains the `ModelEvaluation` class for evaluating models, with functions to extract, sort, and handle metric data.
- **model_selection.py**: Offers the `ModelSelection` class for selecting the best models, including functionalities for model linking.
- **model_plotting.py**: Includes the `ModelPlotter` class for visualizing model performance metrics through various plots.
- **latex_template.py**: Features the `LatexTemplate` class, providing functionalities for LaTeX template manipulation, including image and table generation, translation, and significant digit rounding for LaTeX document creation.
- **optuna_plotting.py**: Includes the `OptunaPlotter` class for visualizing Optuna optimization results, with functions for plotting parameter importance, optimization history, and parallel coordinates.

- **main.py**: This file contains the `PaperPrep` class. Its job is to bring together other parts of the program to find and pick the best models. If there are any, it makes charts using Optuna. Then, it creates figures and tables in LaTeX format showing information like how the models performed on the test dataset, their settings, and training graphs, along with the Optuna charts. All these are put together in one LaTeX file, which can be used in your research paper or presentation.
