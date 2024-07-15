from __future__ import annotations

import logging
import os
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from src.evaluation import DatasetStateContainer
from src.models import BaseSDFModel
from src.models import build_simple_neural_net
from src.models.util import build_transformer_model
from src.util import get_adam

Adam = get_adam()


class AbstractEpochEndStrategy(ABC):
    """A base class for strategies to end epochs."""

    @abstractmethod
    def should_end_epoch(self, step: int, loss: float) -> bool:
        raise NotImplementedError


class FixedStepsPerEpochStrategy(AbstractEpochEndStrategy):
    """A strategy that ends an epoch after a fixed number of steps."""

    def __init__(self, steps_per_epoch: int):
        assert (
            isinstance(steps_per_epoch, int) and steps_per_epoch > 0
        ), "Steps per current_step must be a positive integer!"
        self.steps_per_epoch = steps_per_epoch

    def should_end_epoch(self, step: int, loss: float) -> bool:
        return step >= self.steps_per_epoch


class ConvergenceLossStrategy(AbstractEpochEndStrategy):
    """Ends an epoch when the loss converged."""

    def __init__(
        self,
        patience: int = 100,
        min_rel_improvement: float = 0.2 / 100,
        min_abs_improvement: float = 0.0,
    ):
        assert (
            isinstance(patience, int) and patience > 0
        ), "Patience must be a positive integer!"
        self.patience = patience

        # Can NOT be initialized as np.nan!
        self._best_loss = np.inf
        self._patience_counter = 0

        self._min_abs_improvement = float(min_abs_improvement)
        self._min_rel_improvement = float(min_rel_improvement)

    def _is_better_than_best_loss(
        self,
        loss: float,
    ) -> bool:  # Check if loss is better than the best loss
        if (
            loss
            < self._best_loss * (1 - self._min_rel_improvement)
            - self._min_abs_improvement
        ):
            self._best_loss = loss
            self._patience_counter = 0
            return True

        # Otherwise, increment the patience counter
        self._patience_counter += 1
        return False

    def should_end_epoch(self, step: int, loss: float) -> bool:
        self._is_better_than_best_loss(loss)

        if self._patience_counter >= self.patience:
            return True

        return False


class EndOnFirstEpochStrategy(AbstractEpochEndStrategy):
    """Ends the epoch if any of the strategies says so."""

    def __init__(self, strategies: Iterable):
        self.strategies = strategies
        assert all(isinstance(s, AbstractEpochEndStrategy) for s in strategies)

    def should_end_epoch(self, step: int, loss: float) -> bool:
        return any(
            strategy.should_end_epoch(step, loss)
            for strategy in self.strategies
        )


class EndOnLastEpochStrategy(AbstractEpochEndStrategy):
    """Ends the epoch if all of the strategies said so."""

    def __init__(self, strategies: Iterable):
        self.strategies: list = list(strategies)
        assert all(isinstance(s, AbstractEpochEndStrategy) for s in strategies)

    def should_end_epoch(self, step: int, loss: float) -> bool:
        for strategy in self.strategies:
            if strategy.should_end_epoch(step, loss):
                # Remove the strategy from the list
                self.strategies.remove(strategy)

        return len(self.strategies) == 0


class AdversarialSDF(BaseSDFModel):
    """An adversarial asset pricing model (Pelger et al)."""

    current_sharpe: tf.Tensor
    """The current Sharpe ratio of the model, updated during training."""

    def __init__(
        self,
        num_moments: int = 8,
        print_every_n_steps: int = 10,
        save_checkpoint: bool = True,
    ):
        # Set up required models and optimizers
        self.num_moments = num_moments
        self._print_every_n_steps = int(print_every_n_steps)
        self.train = None
        self._sdf_model = None
        self._moment_model = None
        self._sdf_optimizer = None
        self._moment_optimizer = None
        self.current_sharpe = None
        self.save_checkpoint = save_checkpoint

        # Initialize results folder based on the class name, only if save_checkpoints is True
        if self.save_checkpoint:
            base_path = Path(__file__).parent / "model_checkpoints"
            self.results_folder = base_path / self.__class__.__name__
            self.results_folder.mkdir(parents=True, exist_ok=True)
        else:
            self.results_folder = (
                None  # or any default value or behavior you prefer
            )

        self.results_folder = self._create_results_folder()

    def _create_results_folder(self):
        if not self.save_checkpoint:
            return None

        # Define the base path
        base_path = Path(__file__).parents[2] / "model_checkpoints"

        # Get the current date
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Create a folder for the current date
        date_folder = base_path / date_str
        date_folder.mkdir(exist_ok=True, parents=True)

        # Check for existing run folders and determine the next run number
        existing_runs = [
            d
            for d in date_folder.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]
        run_numbers = [int(d.name.split("_")[1]) for d in existing_runs]
        next_run_number = max(run_numbers, default=0) + 1

        # Create a new run folder
        run_folder = date_folder / f"run_{next_run_number}"
        run_folder.mkdir(exist_ok=True)

        return run_folder

    def get_sdf_weights(self, dataset: DatasetStateContainer) -> np.ndarray:
        self.features = tf.constant(dataset.features, dtype=tf.float32)
        self.mask = tf.constant(dataset.mask, dtype=tf.float32)

        sdf_weights = self._compute_sdf_weights()

        return sdf_weights

    def fit(
        self,
        train: DatasetStateContainer,
        sdf_end_strategy: AbstractEpochEndStrategy = ConvergenceLossStrategy(),
        moment_end_strategy: AbstractEpochEndStrategy = ConvergenceLossStrategy(),
        n_epochs: int = 5,
    ):
        self._init_models()
        self._init_optimizers()
        self.train = train
        self._init_tensors_for_compiled_training()

        self.training_start_time = pd.Timestamp.now()

        for e in range(n_epochs):
            # New optimizer for each epoch!
            # This doesn't make the losses jump so hard
            # self._init_optimizers()
            logging.info(f"Training epoch: {e}")

            try:
                self.train_model_with_strategy(
                    update_sdf=True,
                    end_strategy=deepcopy(sdf_end_strategy),
                )

                if not e == n_epochs - 1:
                    # In the last epoch, we don't need to fit the moment model again
                    # Create a new moment network
                    # self._init_moment_net()
                    self.train_model_with_strategy(
                        update_sdf=False,
                        end_strategy=deepcopy(moment_end_strategy),
                    )

                self._save_checkpoint(epoch=e)

                # Estimate how much time is left in training
                time_elapsed: pd.Timedelta = (
                    pd.Timestamp.now() - self.training_start_time
                )
                logging.info(80 * "=")
                logging.info(80 * "=")
                logging.info(f"Finished training epoch: {e}")
                time_per_epoch = time_elapsed / (e + 1)
                epochs_left = n_epochs - e - 1
                time_left: pd.Timedelta = time_per_epoch * epochs_left
                estimated_end_time = pd.Timestamp.now() + time_left

                logging.info(f"Time per epoch: {time_per_epoch}")
                logging.info(f"Estimated time left: {time_left}")
                logging.info(
                    "Estimated end time: {}".format(
                        estimated_end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
                logging.info(80 * "=")
                logging.info(80 * "=")

            except KeyboardInterrupt:
                logging.info("Training interrupted by user!")
                logging.info(f"Trained until epoch: {e}")
                break

        # We do NOT want to keep a reference to the train set in this class
        self.train = None

        return self

    def _save_checkpoint(self, epoch: int):
        if self.results_folder is not None:
            checkpoint_path = (
                self.results_folder
                / f"{self.__class__.__name__}_epoch_{epoch:02d}.h5"
            )
            # Example saving code (replace with your actual saving logic)
            self._sdf_model.save(checkpoint_path)
        else:
            logging.warning(
                "results_folder is not defined. Checkpoint not saved.",
            )

    def model_factory(self, num_output_units: int) -> keras.Model:
        return build_simple_neural_net(
            output_units=num_output_units,
            add_replace_nan_layer=True,
        )

    def _init_models(self):
        self._sdf_model = self.model_factory(num_output_units=1)
        self._init_moment_net()

    def _init_moment_net(self):
        self._moment_model = self.model_factory(
            num_output_units=self.num_moments,
        )
        self._moment_optimizer = Adam()

    def _init_optimizers(self):
        self._sdf_optimizer = Adam()
        self._moment_optimizer = Adam()

    def train_model_with_strategy(
        self,
        update_sdf: bool,
        end_strategy: AbstractEpochEndStrategy,
    ) -> None:
        assert self.train is not None, "Can only be used in fit() method!"
        logging.info(
            "Training {} model".format("SDF" if update_sdf else "moment"),
        )
        step: int = 0
        loss: float = np.nan

        # Do some pre-computations
        if update_sdf:
            # Pre-compute the moments and update the SDF while training only
            self._moments = tf.constant(self._compute_moments())
        else:
            # Pre-compute the SDF weights and update the moments while training only
            self._sdf_weights = tf.constant(self._compute_sdf_weights())

        # We set some hard max on the steps, in case the strategy has some kind of bug
        while step < 100_000:
            loss: float = self._train_model_for_one_step(step, update_sdf)
            end: bool = end_strategy.should_end_epoch(
                step,
                loss if update_sdf else 1.0 / loss,
            )

            if end:
                break

            step += 1

    def _train_model_for_one_step(
        self,
        current_step: int,
        update_sdf: bool,
    ) -> float:
        with tf.GradientTape() as tape:
            if update_sdf:
                # Compute the SDF weights in the graph
                self._sdf_weights = self._compute_sdf_weights()
            else:
                self._moments = self._compute_moments()

            # Adjust this for training
            sdf_loss, sharpe = self._compute_adversarial_loss(
                self._sdf_weights,
                self._moments,
            )
            moment_loss = -sdf_loss

        # Don't overkill it!
        if (
            current_step % self._print_every_n_steps == 0
            and tf.executing_eagerly()
        ):
            logging.info(
                "Step {}: Loss: {:.1e}, Sharpe: {:.4f}".format(
                    current_step,
                    sdf_loss.numpy(),
                    sharpe.numpy(),
                ),
            )
        # Now update the parameters
        if update_sdf:
            sdf_gradients = tape.gradient(
                sdf_loss,
                self._sdf_model.trainable_variables,
            )
            self._sdf_optimizer.apply_gradients(
                zip(sdf_gradients, self._sdf_model.trainable_variables),
            )

        else:
            # Update moment net params
            moment_gradients = tape.gradient(
                moment_loss,
                self._moment_model.trainable_variables,
            )
            self._moment_optimizer.apply_gradients(
                zip(moment_gradients, self._moment_model.trainable_variables),
            )

        return float(sdf_loss)

    def _compute_moments(self) -> tf.Tensor:
        moments = self._moment_model(self.features)
        moments = tf.nn.tanh(moments)

        return moments

    def _compute_sdf_weights(self) -> tf.Tensor:
        sdf_weights = self._sdf_model(self.features)[..., 0]

        return sdf_weights

    def _init_tensors_for_compiled_training(self, *args, **kwargs) -> None:
        """Initialize the tensors for the compiled training loop."""

        self.mask = tf.constant(self.train.mask, dtype=tf.float32)
        self.returns_with_zeros = tf.constant(
            self.train.returns_with_zeros,
            dtype=tf.float32,
        )
        self.T_i = tf.constant(self.train.T_i, dtype=tf.float32)
        self.num_time_steps = tf.constant(
            self.train.num_time_steps,
            dtype=tf.float32,
        )
        self.num_assets = tf.constant(self.train.num_assets, dtype=tf.float32)
        self.features = tf.constant(self.train.features, dtype=tf.float32)

    @tf.function
    def _compute_adversarial_loss(
        self,
        sdf_weights: tf.Tensor,
        moments: tf.Tensor,
    ) -> tuple:
        # Compute the adversarial pricing errors
        masked_sdf_weights = sdf_weights * self.mask
        abs_weight_per_period = tf.reduce_sum(
            tf.abs(masked_sdf_weights),
            axis=1,
        )
        normalized_weights = masked_sdf_weights / abs_weight_per_period[:, None]
        # Compute F_t (sdf portfolio return) and the SDF's value itself
        sdf_portfolio_return = tf.reduce_sum(
            normalized_weights * self.returns_with_zeros,
            axis=1,
        )

        current_sharpe = tf.reduce_mean(
            sdf_portfolio_return,
        ) / tf.math.reduce_std(sdf_portfolio_return)

        m_t = 1 - sdf_portfolio_return
        # Compute the term inside the pricing error loss

        # shape: (n_time_steps, n_assets, n_moments)
        inside_expectation = (
            moments * self.returns_with_zeros[..., None] * m_t[:, None, None]
        )

        # shape: (n_assets, n_moments)
        expected_pricing_error = (
            tf.reduce_sum(inside_expectation, axis=0) / self.T_i[..., None]
        )
        squared_loss = (
            tf.square(expected_pricing_error)
            * self.T_i[..., None]
            / self.num_time_steps
        )
        loss_per_test_asset = (
            tf.reduce_sum(
                squared_loss,
                axis=0,
            )
            / self.num_assets
        )
        final_loss = tf.reduce_mean(loss_per_test_asset)
        return final_loss, current_sharpe


class TransformerAdversarialSDF(AdversarialSDF):
    """A transformer based adversarial asset pricing model."""

    # Actual number is 7141, but we want to be on the safe side
    _max_assets: int = 7200
    """Max number of assets the transformer can handle."""

    def model_factory(self, num_output_units: int) -> keras.Model:
        return build_transformer_model(
            output_units=num_output_units,
            add_replace_nan_layer=True,
            num_assets=self._max_assets,
        )

    def _init_tensors_for_compiled_training(self, *args, **kwargs) -> None:
        super()._init_tensors_for_compiled_training(*args, **kwargs)

        # But now actually pad the features as well as the asset mask
        # to make it usable with the model

        # Problem: tensors too big!
        # OOM when allocating tensor with shape[240,7200,7200] and type float
        self.features = tf.pad(
            self.features,
            [[0, 0], [0, self._max_assets - self.num_assets], [0, 0]],
        )
        self.mask = tf.pad(
            self.mask,
            [[0, 0], [0, self._max_assets - self.num_assets]],
        )

    def _compute_sdf_weights(self) -> tf.Tensor:
        sdf_weights = self._batched_prediction(self._sdf_model)

        return sdf_weights

    def _compute_moments(self) -> tf.Tensor:
        moments = self._batched_prediction(self._moment_model)

        return moments

    def _batched_prediction(self, model, batch_size: int = 5) -> tf.Tensor:
        y = []
        n_total_steps: int = int(self.num_time_steps.numpy())

        for batch_start in range(0, n_total_steps, batch_size):
            batch_end = min(batch_start + batch_size, n_total_steps)
            batch_features = self.features[batch_start:batch_end]
            batch_mask = self.mask[batch_start:batch_end]

            y_hat = model((batch_features, batch_mask))

            # And unpad the y_hat's
            unpadded_y_hat = y_hat[:, : int(self.num_assets), :]

            y.append(unpadded_y_hat)

        final_y = tf.concat(y, axis=0)
        return final_y


class AdversarialSDFWithPenaltyLoss(AdversarialSDF):
    """An adversarial asset pricing model with penalty loss."""

    #    @tf.function
    def _compute_adversarial_loss(
        self,
        sdf_weights: tf.Tensor,
        moments: tf.Tensor,
    ) -> tuple:
        """
        Compute the adversarial loss with penalty loss.
        """

        # Compute the adversarial pricing errors
        masked_sdf_weights = sdf_weights * self.mask
        abs_weight_per_period = tf.reduce_sum(
            tf.abs(masked_sdf_weights),
            axis=1,
        )
        normalized_weights = masked_sdf_weights / abs_weight_per_period[:, None]
        # Compute F_t (sdf portfolio return) and the SDF's value itself
        sdf_portfolio_return = tf.reduce_sum(
            normalized_weights * self.returns_with_zeros,
            axis=1,
        )

        current_sharpe = tf.reduce_mean(
            sdf_portfolio_return,
        ) / tf.math.reduce_std(sdf_portfolio_return)

        m_t = 1 - sdf_portfolio_return
        # Compute the term inside the pricing error loss

        # shape: (n_time_steps, n_assets, n_moments)
        inside_expectation = (
            moments
            * self.returns_with_zeros[..., None]
            * ((self.features[..., 42, None] + 3.0) * self.mask[..., None])
            * m_t[:, None, None]
        )

        # shape: (n_assets, n_moments)
        expected_pricing_error = (
            tf.reduce_sum(inside_expectation, axis=0) / self.T_i[..., None]
        )
        squared_loss = (
            tf.square(expected_pricing_error)
            * self.T_i[..., None]
            / self.num_time_steps
        )
        loss_per_test_asset = (
            tf.reduce_sum(
                squared_loss,
                axis=0,
            )
            / self.num_assets
        )

        final_loss = tf.reduce_mean(loss_per_test_asset)
        return final_loss, current_sharpe
