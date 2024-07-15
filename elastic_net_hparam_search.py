"""
Estimates and evaluates a linear asset pricing model.

Computes statistics and saves plots to a folder.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm

from estimation import build_datasets
from src.data_loading import load_equity_data
from src.estimation_util import camel_case_to_underscore
from src.estimation_util import plot_cumulative_decile_portfolio_returns
from src.estimation_util import plot_equity_curves
from src.evaluation import DatasetStateContainer
from src.models import ElasticNetSDFModel
from src.models import LinearRegressionBetaModel
from src.models import LinearSDFModel
from src.models import RidgeSDFModel

# Store the results in this folder
folder: Path = Path(__file__).parent / "results"
folder.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def plot_sharpe_ratios(results_df: pd.DataFrame) -> plt.Figure:
    # Create a grid for interpolation
    xi = np.logspace(
        np.log10(results_df["lambda_1"].min()),
        np.log10(results_df["lambda_1"].max()),
        100,
    )
    yi = np.logspace(
        np.log10(results_df["lambda_2"].min()),
        np.log10(results_df["lambda_2"].max()),
        100,
    )
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the Sharpe ratios
    zi = griddata(
        (results_df["lambda_1"], results_df["lambda_2"]),
        results_df["valid_sharpe"],
        (xi, yi),
        method="linear",
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("Sharpe Ratios for Elastic Net SDF Model")
    ax.set_xlabel("Lambda 1")
    ax.set_ylabel("Lambda 2")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Plot the interpolated data
    im = ax.contourf(xi, yi, zi, levels=100, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Sharpe Ratio")

    return fig


def main():
    sets, train = build_datasets()
    valid = sets[1]

    alphas = np.geomspace(start=1e-6, stop=1e-4, num=100)
    results = defaultdict(list)

    for alpha in alphas:
        sdf_model = RidgeSDFModel(alpha=alpha).fit(train)
        train_sharpe = sdf_model.score(train)
        valid_sharpe = sdf_model.score(valid)
        results["alpha"].append(alpha)
        results["valid_sharpe"].append(valid_sharpe)
        results["train_sharpe"].append(train_sharpe)

    # Scatter plot of the results
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    axs.set_title("Validation Sharpe Ratios for Ridge SDF Model")

    axs.set_xlabel("Alpha")
    axs.set_ylabel("Sharpe Ratio")
    axs.set_xscale("log")
    axs.scatter(
        results["alpha"],
        results["valid_sharpe"],
        label="Valid Sharpe",
        color="blue",
    )
    axs.scatter(
        results["alpha"],
        results["train_sharpe"],
        label="Train Sharpe",
        color="red",
    )
    axs.legend()
    fig.savefig(folder / "ridge_sdf_sharpe.png")
    plt.close(fig)

    logging.info(
        "Ridge SDF Model: Best Alpha: %.4e",
        results["alpha"][np.argmax(results["valid_sharpe"])],
    )


if __name__ == "__main__":
    main()
