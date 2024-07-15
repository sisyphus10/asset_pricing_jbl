"""
Where all models and model-related code is stored and organized.
"""

from __future__ import annotations

import logging
import time
from abc import ABC
from abc import abstractmethod
from functools import reduce
from typing import Optional

import keras
import numpy as np
import scipy
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from src.evaluation import DatasetStateContainer
from src.models.base import BaseBetaModel
from src.models.base import BaseSDFModel
from src.models.base import BootstrappedSDFModel
from src.models.util import build_simple_neural_net
from src.models.util import relu


class EnsembleFFNKeras(BaseEstimator):
    def __init__(self, n_models=5, n_epochs=10):
        self.n_models = n_models
        self.n_epochs = n_epochs
        self.models = []

    def fit(self, X, y):
        self.models = []

        for _ in tqdm(range(self.n_models), desc="Fitting models"):
            model = build_simple_neural_net()
            model.fit(
                x=X,
                y=y,
                epochs=self.n_epochs,
                batch_size=1024,
                shuffle=True,
                verbose=0,
                validation_split=0.2,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=2),
                ],
            )
            self.models.append(model)

        return self

    def predict(self, X):
        preds = [
            model.predict(X, batch_size=4096, verbose=0)
            for model in self.models
        ]

        return np.mean(preds, axis=0)


class PointwiseSDFMixin(BaseSDFModel, ABC):
    """Can support multiple point wise SDF models."""

    _sdf_model: keras.Model

    def get_sdf_weights(self, dataset: DatasetStateContainer) -> np.ndarray:
        sdf_weights = self._sdf_model.predict(
            dataset.flattened_features,
            batch_size=2048,
        )

        return sdf_weights


class LinearSDFMixin(BaseSDFModel, ABC):
    """A mixin for linear SDF models."""

    _gamma_param: np.ndarray | None
    """The linear regression coefficient vector."""

    _double_features: bool = True
    """Whether to split features and add ReLUs from zero."""

    _add_constant: bool = False
    """Whether to add a constant to the features or not."""

    _subtract_one_half: bool = False
    """Whether to subtract 0.5 from the features or not."""

    def __init__(
        self,
        double_features: bool = True,
        add_constant: bool = False,
        subtract_one_half: bool = False,
    ):
        self._gamma_param = None
        self._double_features = double_features
        self._add_constant = add_constant
        self._subtract_one_half = subtract_one_half

    @property
    def fitted_with_n_features(self) -> int:
        return len(self._gamma_param)

    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        feats: np.ndarray = np.copy(features)

        if self._subtract_one_half:
            feats = feats - 0.5

        if self._double_features:
            feats = np.concatenate([relu(feats), -relu(-feats)], axis=-1)

        if self._add_constant:
            feats = np.concatenate(
                [feats, np.ones_like(feats[..., :1])],
                axis=-1,
            )

        return feats

    def get_sdf_weights(self, dataset: DatasetStateContainer) -> np.ndarray:
        # Preprocess the flattened features
        feats = self._preprocess_features(dataset.flattened_features)

        assert (
            feats.shape[-1] == self.fitted_with_n_features
        ), "(Preprocessed) features have wrong shape!"

        return feats @ self._gamma_param  # w_{LS} = I_{t,i} @ gamma

    def _get_f_tilde_from_dataset(
        self,
        dataset: DatasetStateContainer,
    ) -> np.ndarray:
        preprocessed_feats = self._preprocess_features(
            dataset.flattened_features,
        )
        r = dataset.flattened_returns[:, None]
        tilde = preprocessed_feats * r  # I_{t,i} * R_{t+1,i}
        tilde = np.stack(
            [
                np.mean(f, axis=0)  # Average over N
                for f in np.split(tilde, dataset.idxs, axis=0)
                if len(f) > 0
            ],  # F_{t+1}
        )
        f_tilde = tilde
        return f_tilde

    def _estimate_linear_sdf(self, dataset) -> np.ndarray:
        f_tilde = self._get_f_tilde_from_dataset(dataset)
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


class LinearSDFModel(LinearSDFMixin):
    """The 'OLS' estimate for the SDF weights."""

    name: str = "LeastSquaresSDFModel"

    def fit(self, dataset: DatasetStateContainer):
        # Estimate the linear model
        self._gamma_param = self._estimate_linear_sdf(dataset)

        assert self._gamma_param.ndim == 1
        return self


class RidgeSDFModel(LinearSDFMixin):
    """A ridge estimate for the SDF weights."""

    def __init__(self, alpha: float = 1.3530e-05, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def fit(self, dataset: DatasetStateContainer):
        # Estimate the linear model
        f_tilde: np.ndarray = self._get_f_tilde_from_dataset(dataset)
        f_tilde_mean = f_tilde.mean(axis=0)
        f_tilde_cov = f_tilde.T @ f_tilde

        # define RHS and LHS of the ridge regression
        lhs = f_tilde_cov + self.alpha * np.eye(f_tilde_cov.shape[0])
        rhs = f_tilde_mean

        # Solve the ridge regression
        self._gamma_param = np.linalg.solve(lhs, rhs)

        assert self._gamma_param.ndim == 1
        return self


class ElasticNetSDFModel(LinearSDFMixin):
    """Kozak and Nagels' Elastic Net SDF model."""

    name: str = "ElasticNetSDFModel"

    def __init__(
        self,
        lambda_1: float = 0.1,
        lambda_2: float = 0.1,
        double_features=True,
        add_constant=False,
        subtract_one_half=False,
    ):
        super().__init__(
            double_features=double_features,
            add_constant=add_constant,
            subtract_one_half=subtract_one_half,
        )
        assert lambda_1 >= 0.0 and isinstance(
            lambda_1,
            float,
        ), "Lambda_1 must be a non-negative float!"
        assert lambda_2 >= 0.0 and isinstance(
            lambda_2,
            float,
        ), "Lambda_2 must be a non-negative float!"

        if lambda_1 == 0.0 and lambda_2 == 0.0:
            logging.warning(
                "Both lambda_1 and lambda_2 are zero. This is equivalent to the OLS model.",
            )

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self._gamma_param = None

    def fit(self, dataset: DatasetStateContainer):
        # Estimate Elastic Net SDF model
        f_tilde: np.ndarray = self._get_f_tilde_from_dataset(dataset)
        f_tilde_mean = f_tilde.mean(axis=0)
        f_tilde_cov = f_tilde.T @ f_tilde

        def objective(theta: np.ndarray) -> float:
            rss = np.mean(np.square(f_tilde_mean - f_tilde_cov @ theta))
            l1_norm = np.linalg.norm(theta, ord=1)
            l2_norm = np.linalg.norm(theta, ord=2) ** 2
            penalty = self.lambda_1 * l1_norm + self.lambda_2 * l2_norm

            return penalty + rss

        # Take LS estimate as starting point
        theta_ls = self._estimate_linear_sdf(dataset)

        start = time.time()
        theta_en = scipy.optimize.minimize(objective, x0=theta_ls).x
        print(f"Elastic Net estimation took {time.time() - start:.2f} seconds")

        self._gamma_param = theta_en

        assert self._gamma_param.ndim == 1
        return self


class ReturnPredictionSDFMixin(BaseSDFModel):
    """A mixin that helps implement fitting methods for multiple models."""

    model: BaseEstimator

    @abstractmethod
    def model_factory(self):
        """A factory method for instantiating a new model to be fit."""

        raise NotImplementedError

    def fit(self, dataset: DatasetStateContainer):
        self.model = self.model_factory()

        self.model.fit(
            X=dataset.flattened_features,
            y=dataset.flattened_returns,
        )

        return self

    def get_sdf_weights(self, dataset: DatasetStateContainer) -> np.ndarray:
        return self.model.predict(dataset.flattened_features)


class EnsembleFFNSDFModel(ReturnPredictionSDFMixin):
    """An ensemble of neural nets to estimate the SDF weights."""

    models: list

    def __init__(self, n_models: int = 10, n_epochs: int = 5):
        self.n_models = n_models
        self.n_epochs = n_epochs

    def model_factory(self):
        return EnsembleFFNKeras(self.n_models, self.n_epochs)


class LassoCVSDFModel(ReturnPredictionSDFMixin):
    """Lasso regression to estimate the SDF weights."""

    def model_factory(self):
        return LassoCV(cv=5)


class BootstrappedLinearSDFModel(BootstrappedSDFModel):
    """Many linear SDF models."""

    _gamma_matrix: np.ndarray
    """All the gamma parameters of the models stacked."""

    models: list[LinearSDFModel]

    def __init__(self, n_models: int = 500, bagging_percentage: float = 0.15):
        super().__init__(
            base_sdf_model=LinearSDFModel(),
            n_models=n_models,
            bagging_percentage=bagging_percentage,
            aggregation_method="median",
        )

    def fit(self, train: DatasetStateContainer, **fit_kwargs):
        super().fit(train, **fit_kwargs)

        self._gamma_matrix = np.column_stack(
            [model._gamma_param for model in self.models],
        )

        return self

    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        return self.models[0]._preprocess_features(features)

    def get_sdf_weights(self, dataset: DatasetStateContainer) -> np.ndarray:
        preprocessed_features = self._preprocess_features(
            dataset.flattened_features,
        )
        num_elements: int = reduce(
            lambda x, y: x * y,
            preprocessed_features.shape,
        )

        if num_elements < 100_000:
            sdf_weights = self._get_predictions(preprocessed_features)
        else:
            batch_size: int = 2**14
            sdf_weights = []

            for idx in tqdm(
                range(0, preprocessed_features.shape[0], batch_size),
                desc="Making predictions",
                smoothing=0.0,
            ):
                batch = preprocessed_features[idx : idx + batch_size]
                sdf_weights.append(self._get_predictions(batch))

            sdf_weights = np.concatenate(sdf_weights)

        return sdf_weights

    def _get_predictions(self, preprocessed_features):
        predictions = preprocessed_features @ self._gamma_matrix
        sdf_weights = np.median(predictions, axis=1)
        return sdf_weights


class SKLearnBetaModelMixin(BaseBetaModel, ABC):
    """A beta-estimation model based on an SKLearn estimator."""

    _sklearn_model: BaseEstimator
    """The (fitted) estimator following the SKLearn API."""

    @abstractmethod
    def model_factory(self) -> BaseEstimator:
        raise NotImplementedError

    def fit(self, dataset: DatasetStateContainer):
        self._sklearn_model = self.model_factory()

        flattened_beta_targets = self._get_flattened_beta_targets(dataset)
        self._sklearn_model.fit(
            dataset.flattened_features,
            flattened_beta_targets,
        )

        return self

    @property
    def is_fitted(self) -> bool:
        if self._sklearn_model is None:
            return False

        if hasattr(self._sklearn_model, "is_fitted"):
            return self._sklearn_model.is_fitted

        return True

    def estimate_risk_loadings(
        self,
        dataset: DatasetStateContainer,
    ) -> np.ndarray:
        assert self.is_fitted, "Model must be fitted first!"

        return self._sklearn_model.predict(dataset.flattened_features)


class LinearRegressionBetaModel(SKLearnBetaModelMixin):
    """OLS regression for beta estimation."""

    name: str = "OLSBetaModel"
    _sklearn_model: LinearRegression | None

    def __init__(self, fit_intercept: bool = False):
        super().__init__()
        self._fit_intercept = bool(fit_intercept)
        self._sklearn_model = None

    def model_factory(self) -> BaseEstimator:
        return LinearRegression(fit_intercept=self._fit_intercept)


class LassoCVBetaModel(SKLearnBetaModelMixin):
    """Lasso regression for beta estimation."""

    name: str = "LassoCVBetaModel"
    _sklearn_model: LassoCV | None

    def __init__(self, cv: int = 5):
        super().__init__()
        self._cv = int(cv)
        self._sklearn_model = None

    def model_factory(self) -> BaseEstimator:
        return LassoCV(cv=self._cv)


class RidgeCVBetaModel(SKLearnBetaModelMixin):
    """Ridge regression for beta estimation."""

    name: str = "RidgeCVBetaModel"
    _sklearn_model: RidgeCV | None

    def __init__(self, cv: int = 5):
        super().__init__()
        self._cv = int(cv)
        self._sklearn_model = None

    def model_factory(self) -> BaseEstimator:
        return RidgeCV(cv=self._cv)


class PLSRegressionBetaModel(SKLearnBetaModelMixin):
    """Partial Least Squares regression for beta estimation."""

    name: str = "PLSRegressionBetaModel"
    _sklearn_model: PLSRegression | None

    def __init__(self, n_components: int = 10):
        super().__init__()
        self._n_components = int(n_components)
        self._sklearn_model = None

    def model_factory(self) -> BaseEstimator:
        return PLSRegression(n_components=self._n_components)


class PLSRegressionCVBetaModel(SKLearnBetaModelMixin):
    """Runs CV for number of components in PLSRegression."""

    def __init__(self):
        super().__init__()
        self._sklearn_model = None

    def model_factory(self) -> BaseEstimator:
        return GridSearchCV(
            PLSRegression(),
            param_grid={"n_components": np.arange(1, 21)},
            cv=5,
        )


class EnsembleFFNBetaModel(SKLearnBetaModelMixin):
    """An ensemble of neural nets to estimate the betas wrt an SDF."""

    def __init__(self, n_models=5, n_epochs=10):
        self.n_models = n_models
        self.n_epochs = n_epochs

    def model_factory(self) -> BaseEstimator:
        return EnsembleFFNKeras(n_models=self.n_models, n_epochs=self.n_epochs)


class BetaModelFromSDFModel(BaseBetaModel):
    """A beta model that uses the SDF model to estimate the betas.

    Each risk loading is then proportional to the SDF weight of that asset.
    """

    _sdf_model: BaseSDFModel

    def __init__(self, sdf_model: BaseSDFModel):
        self._sdf_model = sdf_model

    def fit(self, dataset: DatasetStateContainer):
        """Model already fitted."""
        return self

    def estimate_risk_loadings(
        self,
        dataset: DatasetStateContainer,
    ) -> np.ndarray:
        return self._sdf_model.get_sdf_weights(dataset)


# TODO: @Johannes - Train GAN model and collect / prepare
# TODO: @Johannes - Script to load / explore data from Daniel
