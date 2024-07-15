import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from numpy import ndarray

from src.data_loading import load_equity_data, load_macro_data
from src.tracker import RunTracker


def missing_function(*args, **kwargs):
    raise NotImplementedError("This function has not been implemented yet")


# Run configuration dictionary
run_config: dict = {}


def reset_run_config() -> None:
    """Resets the run configuration."""
    global run_config
    run_config = dict()

    run_config["Time Started"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_config["Platform"] = os.uname().sysname
    run_config["Username"] = os.getlogin()
    run_config["Python Version"] = sys.version
    run_config["Python Path"] = sys.executable


def set_run_config_param(key: str, value) -> None:
    """Sets a run configuration parameter."""
    global run_config
    if key in run_config:
        logging.warning(f"Overwriting run config key: {key}")

    run_config[key] = value


def get_run_config() -> dict:
    """Returns the run configuration."""
    global run_config
    return run_config


@dataclass(repr=False)
class TrainingConfig:
    """A minimal config for training the asset pricing model."""

    # Has some mixin-like behavior
    load_data: Callable = field(default=missing_function)
    init_models_and_optimizer: Callable = field(default=missing_function)
    compute_metrics: Callable = field(default=missing_function)

    metric_names: list[str] = field(
        default_factory=lambda: [
            "train_loss",
            "train_sharpe",
            "val_loss",
            "val_sharpe",
            "sdf_update",
            "current_step",
            "step",
        ]
    )
    """The names of the metrics to compute"""

    total_epochs: int = 10
    """The total number of current_step to train for. Defaults to 10."""

    steps_per_epoch: int = 100
    """The number of steps per current_step. Defaults to 100."""

    checkpoint_every_n_epochs: int = 1
    """The frequency of saving model checkpoints. Defaults to 1."""

    save_arrays_every_n_epochs: int = 20
    """The frequency of saving computed arrays. Defaults to 20."""

    def __repr__(self):
        repr_attributes = self.as_str_dict()

        return "TrainingConfig({})".format(repr_attributes)

    def as_str_dict(self) -> dict[str, str]:
        attributes = self.__dict__
        repr_attributes = {}
        for key, value in attributes.items():
            if callable(value):
                repr_attributes[key] = f"<function {value.__name__}>"
            else:
                repr_attributes[key] = repr(value)

        return repr_attributes

    @staticmethod
    def checkpoint(tracker: RunTracker, models_and_optimizers: dict) -> None:
        """Saves model checkpoints and computed arrays."""
        for name, model in models_and_optimizers.items():
            try:
                tracker.save_model_checkpoint(model, name)

            except Exception as e:
                logging.error(f"Error saving model {name}: {e}")
