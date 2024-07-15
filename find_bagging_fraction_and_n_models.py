"""
Estimates and evaluates a linear asset pricing model.

Computes statistics and saves plots to a folder.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm

from elastic_net_hparam_search import plot_sharpe_ratios
from estimation import _score_sdf_model_on_datasets
from estimation import build_datasets
from src.data_loading import load_equity_data
from src.estimation_util import camel_case_to_underscore
from src.estimation_util import plot_cumulative_decile_portfolio_returns
from src.estimation_util import plot_equity_curves
from src.estimation_util import set_seeds
from src.evaluation import DatasetStateContainer
from src.models import BaseBetaModel
from src.models import BaseSDFModel
from src.models import BetaModelFromSDFModel
from src.models import BootstrappedLinearSDFModel
from src.models import ElasticNetSDFModel
from src.models import EnsembleFFNBetaModel
from src.models import EnsembleFFNSDFModel
from src.models import LassoCVBetaModel
from src.models import LassoCVSDFModel
from src.models import LinearRegressionBetaModel
from src.models import LinearSDFModel
from src.models import PLSRegressionBetaModel
from src.models import PLSRegressionCVBetaModel
from src.models import RidgeCVBetaModel
from src.models.base import BlockBootstrappedSDFModel
from src.models.base import BootstrappedSDFModel
from src.models.gan_stuff import AdversarialSDF
from src.models.gan_stuff import ConvergenceLossStrategy
from src.models.gan_stuff import EndOnLastEpochStrategy
from src.models.gan_stuff import FixedStepsPerEpochStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Logging setup")


set_seeds()


# Store the results in this folder
folder: Path = Path(__file__).parent / "results"
folder.mkdir(exist_ok=True, parents=True)


def main():
    sets, train = build_datasets()
    valid = sets[1]

    # Choose the SDF estimation logic here!
    # Shouldn't get worse than 0.2% per epoch

    # Try bagged linear SDF model
    results: dict = defaultdict(list)
    bagging_freq_rv = scipy.stats.uniform(0.1, 0.5)
    n_models_rv = scipy.stats.randint(20, 200)

    @print_nothing
    def objective(num_models: int, bagging_frequency: float) -> float:
        sdf_model = BootstrappedLinearSDFModel(
            n_models=num_models,
            bagging_percentage=bagging_frequency,
        ).fit(train)
        valid_sharpe = sdf_model.score(valid)

        return valid_sharpe

    # Try bagged linear SDF model
    for _ in tqdm(range(1_000), desc="Evaluating bagged linear SDF model"):
        num_models = n_models_rv.rvs()
        bagging_frequency = bagging_freq_rv.rvs()
        try:
            valid_sharpe = objective(num_models, bagging_frequency)
            results["num_models"].append(num_models)
            results["bagging_frequency"].append(bagging_frequency)
            results["valid_sharpe"].append(valid_sharpe)

        except KeyboardInterrupt:
            break

    results_df = pd.DataFrame(results)

    # Plot a heatmap of the results
    fig = plot_interpolated_heatmap(
        results_df,
        "num_models",
        "bagging_frequency",
        "valid_sharpe",
        "Sharpe Ratios for Bagged Linear SDF Model",
        "Number of Models",
        "Bagging Fraction",
        x_log_scale=False,
        y_log_scale=False,
    )
    fig.savefig(folder / "bagged_linear_sdf_model_heatmap.png")


def print_nothing(func):
    """Re-routes the stdout temporarily to /dev/null."""
    from contextlib import redirect_stderr, redirect_stdout

    @wraps(func)
    def wrapper(*args, **kwargs):
        with open("/dev/null", "w") as f:
            with redirect_stdout(f) and redirect_stderr(f):
                res = func(*args, **kwargs)

        return res

    return wrapper


def plot_interpolated_heatmap(
    results_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    sharpe_col: str,
    title: str,
    x_label: str,
    y_label: str,
    x_log_scale: bool = True,
    y_log_scale: bool = True,
) -> plt.Figure:
    """
    Plot Sharpe ratios as a contour plot.

    Args:
        results_df: DataFrame with columns for x, y, and Sharpe ratio values.
        x_col: Name of the column containing x-axis values.
        y_col: Name of the column containing y-axis values.
        sharpe_col: Name of the column containing Sharpe ratio values.
        title: Title of the plot.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.

    Returns:
        Matplotlib Figure object.
    """
    # Create a grid for interpolation
    if x_log_scale:
        xi = np.logspace(
            np.log10(results_df[x_col].min()),
            np.log10(results_df[x_col].max()),
            100,
        )
    else:
        xi = np.linspace(results_df[x_col].min(), results_df[x_col].max(), 100)
    if y_log_scale:
        yi = np.logspace(
            np.log10(results_df[y_col].min()),
            np.log10(results_df[y_col].max()),
            100,
        )
    else:
        yi = np.linspace(results_df[y_col].min(), results_df[y_col].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the Sharpe ratios
    zi = griddata(
        (results_df[x_col], results_df[y_col]),
        results_df[sharpe_col],
        (xi, yi),
        method="linear",
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Plot the interpolated data
    im = ax.contourf(xi, yi, zi, levels=100, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Sharpe Ratio")

    return fig


if __name__ == "__main__":
    main()
