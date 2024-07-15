from __future__ import annotations

import typing
from textwrap import dedent
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

UNK: float = -99.99


class DatasetStateContainer:
    """Keeps arrays and state computed from a raw dataset.

    Every model DEPENDENT computation is done OUTSIDE of this class.
    Results are stored as variables in this class so methods can use them.
    """

    returns: np.ndarray
    returns_with_nans: np.ndarray
    returns_with_zeros: np.ndarray
    features: np.ndarray
    mask: np.ndarray
    idxs: np.ndarray
    name: str | None

    # And store the flattened versions of it
    flattened_returns: np.ndarray

    # Private variables used only for internal calculations
    _flattened_sdf_weights: np.ndarray
    sdf_portfolio_return: np.ndarray | None

    @property
    def num_assets(self) -> int:
        return self.features.shape[1]

    @property
    def num_time_steps(self) -> int:
        return self.features.shape[0]

    @property
    def num_features(self) -> int:
        return self.features.shape[2]

    @property
    def T_i(self) -> np.ndarray:
        """Number of time steps each asset is observed."""
        return np.sum(self.mask, axis=0)

    @property
    def N_t(self) -> np.ndarray:
        """Number of assets observed at each time step."""
        return np.sum(self.mask, axis=1)

    def __init__(
        self,
        data: np.ndarray,
        feature_names: np.ndarray,
        name: str | None = None,
    ) -> None:
        self.decile_portfolio_returns: np.ndarray | None = None
        self.n_quantiles: int | None = None
        self.epsilon: np.ndarray | None = None
        self.r_t_hat: np.ndarray | None = None
        self.flattened_epsilon: np.ndarray | None = None
        self._flattened_risk_loadings: np.ndarray | None = None
        self.sdf_portfolio_return: np.ndarray | None = None
        assert isinstance(data, np.ndarray), "data must be a NumPy array"
        assert data.ndim == 3, "data must be a 3D array"

        self.data = data
        self.name = name

        self.returns = data[..., 0]
        self.return_with_nans = np.copy(self.returns)
        self.return_with_nans[self.returns == UNK] = np.nan
        self.returns_with_zeros = np.copy(self.returns)
        self.returns_with_zeros[self.returns == UNK] = 0

        self.features = data[..., 1:]  # - 0.5
        self.feature_names = feature_names

        self.mask = np.array(self.returns != UNK, dtype=bool)
        self.idxs = np.cumsum(np.sum(self.mask, axis=1))

        self.flattened_returns = self.returns[self.mask]
        self.flattened_features = self.features[self.mask]

    # Finding the SDF portfolio
    # Model: (...) -> sdf weights omega_{t, i} (unnormalized)
    def set_sdf_portfolio_weights(self, sdf_weights: np.ndarray) -> None:
        if sdf_weights.ndim == 1:  # Shape (T * N)
            flattened_weights = sdf_weights

        elif sdf_weights.ndim == 2:  # Shape (T, N)
            if sdf_weights.shape[1] == 1:  # Shape (T * N, 1)
                flattened_weights = sdf_weights[:, 0]
            else:
                flattened_weights = sdf_weights[self.mask]

        else:
            raise ValueError("sdf_weights must be a 1D or 2D array")

        # And store the flattened version of it
        self._flattened_sdf_weights = flattened_weights

    def calculate_sdf_portfolio_returns_from_weights(self) -> None:
        sdf_portfolio_return = []

        for weights_in_period, asset_returns_in_period in zip(
            np.split(self._flattened_sdf_weights, self.idxs, axis=0),
            self.splitted_returns(),
        ):
            if len(weights_in_period) == 0:
                continue

            # Normalize the weights
            normalized_weights_in_period = weights_in_period / np.sum(
                np.abs(weights_in_period),
            )

            return_in_period = np.sum(
                normalized_weights_in_period * asset_returns_in_period,
            )

            sdf_portfolio_return.append(return_in_period)

        self.sdf_portfolio_return = np.array(sdf_portfolio_return, dtype=float)

    def splitted_returns(self):
        return np.split(self.flattened_returns, self.idxs, axis=0)

    @property
    def sdf_sharpe_ratio(self) -> float:
        self._check_sdf_portfolio_return_is_computed()

        return np.mean(self.sdf_portfolio_return) / np.std(
            self.sdf_portfolio_return,
        )

    def _check_sdf_portfolio_return_is_computed(self):
        if self.sdf_portfolio_return is None:
            raise ValueError(
                dedent(
                    """
                    SDF portfolio return has not been computed yet.
                    Please provide some SDF weights via the `set_sdf_portfolio_weights` method.
                    And then call `calculate_sdf_portfolio_returns_from_weights` to compute the SDF portfolio return.
                    """,
                ),
            )

    @property
    def sdf_cumulative_return(self) -> np.ndarray:
        self._check_sdf_portfolio_return_is_computed()

        return np.cumsum(self.sdf_portfolio_return) / np.std(
            self.sdf_portfolio_return,
        )

    @property
    def sdf_portfolio_statistics(self) -> pd.Series:
        self._check_sdf_portfolio_return_is_computed()
        d = dict()

        d["mean_return"] = np.mean(self.sdf_portfolio_return)
        d["std_return"] = np.std(self.sdf_portfolio_return)
        d["sharpe_ratio"] = self.sdf_sharpe_ratio
        d["t_statistic"] = d["mean_return"] / (
            d["std_return"] / np.sqrt(self.num_time_steps)
        )
        d["p_value"] = 1 - stats.t.cdf(
            d["t_statistic"],
            self.num_time_steps - 1,
        )
        d["annualized_return"] = np.mean(self.sdf_portfolio_return) * 12
        d["annualized_volatility"] = np.std(
            self.sdf_portfolio_return,
        ) * np.sqrt(12)
        d["annualized_sharpe_ratio"] = (
            d["annualized_return"] / d["annualized_volatility"]
        )

        return self.series_from_dict(d)

    def plot_sdf_portfolio_returns(self) -> None:
        raise NotImplementedError

    def evaluate_sdf_portfolio_performance(self) -> None:
        raise NotImplementedError

    # Finding the risk loadings wrt the SDF
    # Model: (...) -> beta_{t, i} (unnormalized, unscaled)
    def set_risk_loadings(self, beta_hat: np.ndarray) -> None:
        s = beta_hat.shape

        if len(s) == 2 and s[1] == 1:
            b = beta_hat[:, 0]

        elif len(s) == 1:
            b = beta_hat

        else:
            raise ValueError(
                "beta_hat must be a 1D or 2D array, received array of shape: {}".format(
                    s,
                ),
            )

        self._flattened_risk_loadings = b

    def calculate_return_estimates_from_risk_loadings(self) -> None:
        r_t_hat = []
        epsilon = []

        for b, r in self.splitted_beta_return_iterator():
            if len(b) == 0:
                continue

            r_hat = b.dot(r) / b.dot(b) * b
            r_t_hat.append(r_hat)
            epsilon.append(r - r_hat)

        self.r_t_hat = r_t_hat
        self.flattened_epsilon = np.concatenate(epsilon)
        self.epsilon = np.zeros_like(self.returns)
        self.epsilon[self.mask] = self.flattened_epsilon

    def splitted_beta_return_iterator(self):
        return zip(
            np.split(self._flattened_risk_loadings, self.idxs, axis=0),
            self.splitted_returns(),
        )

    def evaluate_pricing_performance(self) -> pd.Series:
        self.check_epsilons_calculated()

        d = dict()

        try:
            d["Monthly Sharpe"] = self.sdf_sharpe_ratio
        except:
            pass

        # Compute normal r2 (explained variation)
        numerator = np.sum(np.square(self.flattened_epsilon))
        denominator = np.sum(np.square(self.flattened_returns))
        d["R2"] = 1 - numerator / denominator

        # Compute cross-sectional r2
        stock_pricing_error = np.sum(self.epsilon, axis=0) / self.T_i
        stock_return_sum = np.nansum(self.return_with_nans, axis=0) / self.T_i
        numerator = np.sum(np.square(stock_pricing_error))
        denominator = np.sum(np.square(stock_return_sum))
        d["XS R2"] = 1 - numerator / denominator

        # Weighted cross-sectional R2
        numerator = np.sum(np.square(stock_pricing_error) * self.T_i)
        denominator = np.sum(np.square(stock_return_sum) * self.T_i)
        d["Weighted XS R2"] = 1 - numerator / denominator

        return self.series_from_dict(d)

    def check_epsilons_calculated(self):
        if self.flattened_epsilon is None:
            raise ValueError(
                dedent(
                    """
                    Pricing performance has not been computed yet.
                    Please provide some risk loadings via the `set_risk_loadings` method.
                    And then call `calculate_return_estimates_from_risk_loadings` to compute the pricing performance.
                    """,
                ),
            )

    def compute_decile_portfolio_returns(self, n_quantiles: int = 10) -> None:
        decile_portfolio_returns = [list() for _ in range(n_quantiles)]
        self.n_quantiles = n_quantiles

        for b, r in self.splitted_beta_return_iterator():
            if len(b) == 0:
                continue

            sort_idxs = np.argsort(b)
            sorted_returns = r[sort_idxs]
            quantiles = np.array_split(sorted_returns, n_quantiles)

            for i, quantile in enumerate(quantiles):
                decile_portfolio_returns[i].append(quantile.mean())

        self.decile_portfolio_returns = np.stack(decile_portfolio_returns)

    @property
    def decile_portfolios_cumulative_returns(self) -> None:
        """Compute the cumulative returns of the beta-sorted decile portfolios.

        Note that the individual decile portfolio returns must already be computed.
        Call `compute_decile_portfolio_returns` beforehand.
        """

        if self.decile_portfolio_returns is None:
            raise ValueError(
                dedent(
                    """
                    Decile portfolio returns have not been computed yet.
                    Please call `compute_decile_portfolio_returns` to compute the decile portfolio returns.
                    """,
                ),
            )

        rets = (
            np.cumsum(self.decile_portfolio_returns, axis=1)
            / np.std(self.decile_portfolio_returns, axis=1)[:, None]
        )

        return rets

    def evaluate_decile_portfolio_performance(self) -> pd.Series:
        d = dict()

        for i in range(self.n_quantiles):
            d["Average Return", i + 1] = self.decile_portfolio_returns[i].mean()

        # And the return of the highest-minus-lowest portfolio
        n_deciles = self.n_quantiles
        d["Average Return", f"{n_deciles}-1"] = (
            d["Average Return", n_deciles] - d["Average Return", 1]
        )

        return self.series_from_dict(d)

    def series_from_dict(self, d):
        if self.name is not None:
            return pd.Series(d, name=self.name)
        return pd.Series(d)

    def sample(self, size: int | float) -> DatasetStateContainer:
        int_size = self._get_integer_size(size)

        idxs = np.random.choice(
            self.num_time_steps,
            size=int_size,
            replace=False,
        )

        return self._subset_from_indices(idxs)

    def _subset_from_indices(self, idxs) -> DatasetStateContainer:
        return DatasetStateContainer(self.data[idxs], self.name)

    def block_sample(
        self,
        size: int | float,
    ) -> DatasetStateContainer:
        int_size = self._get_integer_size(size)

        starting_point = np.random.randint(0, self.num_time_steps - int_size)
        idxs = np.arange(starting_point, starting_point + int_size)

        return self._subset_from_indices(idxs)

    def _get_integer_size(self, size):
        if isinstance(size, int):
            assert (
                0 < size < self.num_time_steps
            ), "size must be an integer between 0 and the number of time steps"
            int_size = size
        elif isinstance(size, float):
            assert 0 < size < 1, "size must be a float between 0 and 1"
            int_size = int(size * self.num_time_steps)
        else:
            raise TypeError("size must be an integer or a float")
        return int_size


class DatasetWithMacro(DatasetStateContainer):
    """A container for datasets that also contain macroeconomic data."""

    macro_data: np.ndarray

    def __init__(
        self,
        data: np.ndarray,
        macro_data: np.ndarray,
        name: str | None = None,
    ) -> None:
        super().__init__(data, name)
        self.macro_data = macro_data


# TODO: Add a container for macro data as a subclass
