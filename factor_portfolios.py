"""
Estimates and evaluates a linear asset pricing model.

Computes statistics and saves plots to a folder.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

from estimation import build_datasets
from src.data_loading import load_equity_data
from src.estimation_util import camel_case_to_underscore
from src.estimation_util import plot_cumulative_decile_portfolio_returns
from src.estimation_util import plot_equity_curves
from src.estimation_util import set_seeds
from src.evaluation import DatasetStateContainer
from src.evaluation import UNK
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
from src.models import RidgeSDFModel
from src.models.base import BlockBootstrappedSDFModel
from src.models.base import BootstrappedSDFModel
from src.models.gan_stuff import AdversarialSDF
from src.models.gan_stuff import ConvergenceLossStrategy
from src.models.gan_stuff import EndOnLastEpochStrategy
from src.models.gan_stuff import FixedStepsPerEpochStrategy
from src.models.gan_stuff import TransformerAdversarialSDF

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Logging setup")


set_seeds()


# Store the results in this folder
folder: Path = Path(__file__).parent / "results"
folder.mkdir(exist_ok=True, parents=True)


def replace_nan(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x[np.isnan(x)] = 0
    x[x == UNK] = 0
    return x


def main():
    sets, train = build_datasets()

    # Build factor portfolios for trainset first
    s = sets[2]  # test set for newest data
    portf_returns = defaultdict(list)
    asset_returns = replace_nan(s.returns_with_zeros)

    for idx, feat_name in enumerate(s.feature_names):
        logging.info(f"Building factor portfolio for {feat_name}")
        feat_values = replace_nan(s.features[:, :, idx])

        factor_portfolio_return = np.sum(feat_values * asset_returns, axis=1)
        portf_returns[feat_name] = factor_portfolio_return

    portf_returns_df = pd.DataFrame(portf_returns)

    # And plot the cumulative returns
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    cum_returns = portf_returns_df.cumsum()
    cum_returns.plot(ax=ax)

    ax.set_title("Cumulative Returns of Factor Portfolios")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Return")
    ax.legend(title="Factor Portfolio")

    fig.savefig(folder / "factor_portfolios_cumulative_returns.png")

    # And calculate some portfolio statistics
    stats = {}

    stats["Mean Monthly Return"] = portf_returns_df.mean()
    stats["Std Dev Monthly Return"] = portf_returns_df.std()
    stats["Monthly Sharpe Ratio"] = (
        stats["Mean Monthly Return"] / stats["Std Dev Monthly Return"]
    )
    stats["t Statistic"] = stats["Monthly Sharpe Ratio"] * np.sqrt(
        s.num_time_steps,
    )
    stats["Absolute t Statistic"] = np.abs(stats["t Statistic"])
    stats["p Value"] = stats["Absolute t Statistic"].apply(
        lambda x: 1 - norm.cdf(x),
    )

    stats_df = pd.DataFrame(stats).sort_values(
        "Absolute t Statistic",
        ascending=False,
    )
    stats_df.to_csv(folder / "factor_portfolios_statistics.csv")
    print(stats_df.round(3))


if __name__ == "__main__":
    main()
