from __future__ import annotations

import logging
import time
from collections.abc import Iterable

import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from src.evaluation import DatasetStateContainer


class ReplaceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, replace_value=-99.99, new_value=0.0):
        self.replace_value = replace_value
        self.new_value = new_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[X == self.replace_value] = self.new_value
        return X_transformed


def plot_equity_curves(
    states: Iterable[DatasetStateContainer],
    axs=None,
    line_plot: bool = False,
) -> plt.Figure:
    fig = None

    if axs is None:
        fig, axs = plt.subplots(figsize=(12, 8))

    idx: int = 0

    for state in states:
        cum_return = state.sdf_cumulative_return
        x = np.arange(idx, idx + len(cum_return))

        if line_plot:
            axs.plot(x, cum_return, label=state.name)
        else:
            axs.scatter(x, cum_return, label=state.name)

        idx += len(cum_return)

    axs.set_title("Equity Curves")
    axs.set_xlabel("Time")
    axs.set_ylabel("Cumulative Return")
    axs.legend()
    # plt.show()

    if fig is not None:
        return fig


def plot_cumulative_decile_portfolio_returns(
    states: Iterable[DatasetStateContainer],
    axs=None,
    line_plot: bool = False,
) -> plt.Figure:
    fig = None

    if axs is None:
        fig, axs = plt.subplots(figsize=(12, 8))

    cmap = plt.get_cmap("tab10")
    idx: int = 0

    for state_idx, state in enumerate(states):
        # shape: (10, T)
        cum_returns = state.decile_portfolios_cumulative_returns

        for i in range(10):
            cum_return = cum_returns[i]
            x = np.arange(idx, idx + len(cum_return))
            # Only add the label for the first state
            label = f"Decile {i + 1}" if state_idx == 0 else None

            if line_plot:
                axs.plot(x, cum_return, label=label, color=cmap(i))
            else:
                axs.scatter(x, cum_return, label=label, color=cmap(i))

        idx += len(cum_return)

    axs.set_title("Cumulative Decile Portfolio Returns")
    axs.set_xlabel("Time")
    axs.set_ylabel("Cumulative Return")
    axs.legend()
    # plt.show()

    if fig is not None:
        return fig


def estimate_elastic_net_sdf(
    train,
    lambda_1: float = 0.1,
    lambda_2: float = 0.1,
) -> np.ndarray:
    # Estimate Elastic Net SDF model
    f_tilde: np.ndarray = get_f_tilde_from_dataset(train)
    f_tilde_mean = f_tilde.mean(axis=0)
    f_tilde_cov = f_tilde.T @ f_tilde

    def objective(theta: np.ndarray) -> float:
        rss = np.mean(np.square(f_tilde_mean - f_tilde_cov @ theta))
        penalty = lambda_1 * np.mean(np.abs(theta)) + lambda_2 * np.mean(
            np.square(theta),
        )

        return penalty + rss

    # Take LS estimate as starting point
    theta_ls = estimate_linear_sdf(train)

    start = time.time()
    theta_en = scipy.optimize.minimize(objective, x0=theta_ls).x
    print(f"Elastic Net estimation took {time.time() - start:.2f} seconds")

    return theta_en


def prepare_beta_targets_for_estimation(train):
    beta_targets = train.returns * train.sdf_portfolio_return[:, None]
    flattened_beta_targets = beta_targets[train.mask]
    return flattened_beta_targets


def estimate_linear_sdf(train: DatasetStateContainer) -> np.ndarray:
    # Estimate the linear model
    f_tilde = get_f_tilde_from_dataset(train)
    f_tilde_mean = f_tilde.mean(axis=0)  # Average over T of F_{t+1}
    # This could, in theory make a difference but apparently it doesn't
    f_tilde_cov = f_tilde.T @ f_tilde
    # f_tilde_cov = np.cov(f_tilde, rowvar=False)
    # Compute the SDF
    # Make sure that it works even if f_tilde_cov is singular
    cov_pinv = np.linalg.pinv(f_tilde_cov)
    theta_ls = cov_pinv @ f_tilde_mean

    # This can only be used if the matrix is non-singular
    # theta_ls = np.linalg.solve(f_tilde_cov, f_tilde_mean)

    return theta_ls


def get_f_tilde_from_dataset(train: DatasetStateContainer) -> np.ndarray:
    """
    Only use the training data to estimate the Theta
    """

    r = train.flattened_returns[:, None]
    f_tilde = train.flattened_features * r  # I_{t,i} * R_{t+1,i}
    f_tilde = np.stack(
        [
            np.mean(f, axis=0)  # Average over N
            for f in np.split(f_tilde, train.idxs, axis=0)
            if len(f) > 0
        ],  # F_{t+1}
    )
    return f_tilde


all_caps: tuple = ("SDF", "FFN", "NN", "SVM", "EN", "PLS", "LS", "CV")


def camel_case_to_underscore(camel: str) -> str:
    """
    Convert a camel case string to underscore separated.
    """
    replaced = camel

    for cap in all_caps:
        l = cap.lower()
        replaced = replaced.replace(cap, f"_{l}")

    return (
        "".join(["_" + c.lower() if c.isupper() else c for c in replaced])
        .lstrip("_")
        .lower()
    )


def set_seeds(seed: int = 42) -> None:
    from random import seed as rseed

    import numpy as np

    rseed(seed)
    np.random.seed(seed)

    try:
        from tensorflow import random as tfrand

        tfrand.set_seed(seed)

    except ImportError:
        logging.warning(
            "Could not import and therefore set seed for tensorflow",
        )

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logging.warning("Could not import and therefore set seed for torch")
