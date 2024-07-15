from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Literal

import numpy as np
from tqdm import tqdm

from src.evaluation import DatasetStateContainer


class BaseSDFModel(ABC):
    """A base class for all models that estimate the SDF weights."""

    name: str

    @abstractmethod
    def fit(self, dataset: DatasetStateContainer):
        raise NotImplementedError

    def score(self, dataset: DatasetStateContainer) -> float:
        """Scores and SDF model on a dataset and returns the monthly sharpe."""

        sdf_weights: np.ndarray = self.get_sdf_weights(dataset)
        dataset.set_sdf_portfolio_weights(sdf_weights)
        dataset.calculate_sdf_portfolio_returns_from_weights()

        return dataset.sdf_sharpe_ratio

    @abstractmethod
    def get_sdf_weights(self, dataset: DatasetStateContainer) -> np.ndarray:
        raise NotImplementedError


class BaseBetaModel(ABC):
    """A base class for all models estimating risk loadings."""

    name: str

    @abstractmethod
    def fit(self, dataset: DatasetStateContainer):
        raise NotImplementedError

    @abstractmethod
    def estimate_risk_loadings(
        self,
        dataset: DatasetStateContainer,
    ) -> np.ndarray:
        raise NotImplementedError

    def _get_flattened_beta_targets(
        self,
        dataset: DatasetStateContainer,
    ) -> np.ndarray:
        beta_targets = (
            dataset.returns * dataset.sdf_portfolio_return[:, None]
        )  # R_{t+1,i} * F_{t=1,i}
        flattened_beta_targets = beta_targets[dataset.mask]

        return flattened_beta_targets

    def score(
        self,
        dataset: DatasetStateContainer,
        n_quantiles: int = 10,
    ) -> None:
        beta_hat = self.estimate_risk_loadings(dataset)

        dataset.set_risk_loadings(beta_hat)
        dataset.calculate_return_estimates_from_risk_loadings()
        dataset.compute_decile_portfolio_returns(n_quantiles=n_quantiles)


class BootstrappedSDFModel(BaseSDFModel):
    """An ensemble of adversarial asset pricing models."""

    models: list[BaseSDFModel]
    """A list of base SDF models."""

    agg_method: Literal["mean", "median"]
    """The method used to aggregate the SDF weights of the models."""

    def __init__(
        self,
        base_sdf_model: BaseSDFModel,
        n_models: int = 20,
        bagging_percentage: float = 0.7,
        aggregation_method: Literal["mean", "median"] = "median",
    ):
        super().__init__()
        self.models = [deepcopy(base_sdf_model) for _ in range(n_models)]
        assert 0.0 < bagging_percentage <= 1.0
        self.bagging_percentage = bagging_percentage
        self.agg_method = aggregation_method

    def fit(self, train: DatasetStateContainer, **fit_kwargs):
        for model in tqdm(self.models, desc="Fitting models", smoothing=0.0):
            bagged_train_set = self._subsample_trainset(train)

            model.fit(bagged_train_set, **fit_kwargs)

        return self

    def _subsample_trainset(
        self,
        train: DatasetStateContainer,
    ) -> DatasetStateContainer:
        # Bagging
        bagged_train_set = train.sample(size=self.bagging_percentage)
        return bagged_train_set

    def get_sdf_weights(self, dataset: DatasetStateContainer) -> np.ndarray:
        iterator = self.models

        if len(iterator) > 10:
            iterator = tqdm(iterator, desc="Getting SDF weights", smoothing=0.0)

        model_sdf_weights = [
            model.get_sdf_weights(dataset) for model in iterator
        ]

        if self.agg_method == "mean":
            sdf_weights = np.mean(model_sdf_weights, axis=0)
        elif self.agg_method == "median":
            sdf_weights = np.median(model_sdf_weights, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.agg_method}")

        return sdf_weights


class BlockBootstrappedSDFModel(BootstrappedSDFModel):
    """Subsamples blocks instead of individual observations."""

    def _subsample_trainset(
        self,
        train: DatasetStateContainer,
    ) -> DatasetStateContainer:
        # Bagging
        bagged_train_set = train.block_sample(size=self.bagging_percentage)
        return bagged_train_set
