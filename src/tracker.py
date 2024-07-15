from __future__ import annotations

import csv
import datetime
import json
from collections import defaultdict
from collections.abc import Iterable
from logging import getLogger
from pathlib import Path
from pprint import pformat
from typing import Optional

import numpy as np

logger = getLogger(__name__)


class RunTracker:
    """
    Tracks and manages a training run, including saving results and model checkpoints to disk.

    This class is designed to handle various aspects of a training run, such as saving training parameters,
    model checkpoints, and other relevant raw_data. It automatically creates a directory for each training run
    and organizes saved files within that directory.

    Attributes:
        path (Path, optional): The file path where the training run raw_data will be saved.
                              If not specified, it defaults to a directory under the user's home directory.
        csv_columns (List[str], optional): Columns to be used for saving CSV raw_data.
        model_counters (defaultdict): A counter for each model being saved, used for naming saved model files.

    Methods:
        save_training_params(params): Saves the training parameters in a JSON file.
        save_model_checkpoint(model, name): Saves a model checkpoint in the specified directory.

    Example:
        run = RunTracker(Path('path/to/save'))
        run.save_training_params({'learning_rate': 0.01, 'batch_size': 32})
        # This saves training parameters to 'training_params.json' in the specified directory.
        run.save_model_checkpoint(model, 'my_model')
        # This saves a model checkpoint in 'model_checkpoints/my_model/' directory.
    """

    def __init__(self, path: Path | None = None):
        """
        Initializes a RunTracker instance with an optional path for saving the training results.

        Args:
            path (Optional[Path]): The directory path for saving training results. If not provided, a new directory
                                   will be created under the user's home directory with a timestamp.
        """

        self.path = path
        self.csv_columns = None

        if self.path is None:
            self.path = Path.home() / "asset-pricing-training-runs/{}".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )

        self.path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving training run results to '{self.path}'")

        self.model_counters = defaultdict(int)

    def save_training_params(self, params: dict):
        """
        Saves the training parameters in a JSON-formatted file for future reference.

        Args:
            params (dict): A dictionary of training parameters to be saved.
        """

        pretty_params = pformat(params)

        with open(self.path / "training_params.json", "w") as f:
            # pprint the dict and save the string representation
            f.write(pretty_params)

    def save_model_checkpoint(self, model, name: str):
        """
        Saves a TensorFlow/Keras model checkpoint to disk.

        Args:
            model (tf.keras.Model): The model to be saved.
            name (str): A name for the model, used for creating a subdirectory under the model checkpoints directory.

        Raises:
            AssertionError: If the 'name' argument is an empty string.
        """
        assert len(name) > 0, "name must be a non-empty string"

        models_dir = self.path / f"model_checkpoints/{name}/"
        models_dir.mkdir(parents=True, exist_ok=True)

        # save the model
        self.model_counters[name] += 1
        model.save(models_dir / f"{self.model_counters[name]:05d}.h5")

    def save_arrays(self, **arrays):
        """
        Saves given arrays in a compressed NumPy file format.

        Args:
            **arrays: Variable number of array-like objects to be saved.
                     Each array is saved under the key provided in the argument.

        This method saves the provided arrays in a NumPy compressed file format (.npz) in a
        directory named 'arrays' under the path specified for this training run. Each file
        is named with an incrementing counter to maintain uniqueness.
        """

        arrays_dir = self.path / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)

        # save the array
        self.model_counters["arrays"] += 1
        np.savez_compressed(
            arrays_dir / "{:05d}.npz".format(self.model_counters["arrays"]),
            **arrays,
        )

    def set_csv_columns(self, columns: Iterable[str]):
        """
        Sets the column names for a CSV file and initializes the file if it doesn't exist.

        Args:
            columns (list): A list of column names to be used in the CSV file.

        This method sets up a CSV file to record various metrics or losses during training.
        It ensures that the column names are unique and includes a date column. If the CSV
        file does not exist, it is created with the specified column headers.
        """

        csv_file = self.path / "losses.csv"
        columns = set(columns)
        columns.add("date")
        # columns = sorted(list(columns))
        columns = list(columns)
        self.csv_columns = columns

        if not csv_file.exists():
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f=f, fieldnames=columns)
                writer.writeheader()

    def save_losses(self, losses: dict):
        """
        Records the given loss metrics in a CSV file.

        Args:
            losses (dict): A dictionary containing loss metrics to be recorded.

        This method appends the given loss metrics to a CSV file specified for the training run.
        It ensures that all the columns specified earlier with 'set_csv_columns' are present
        in the dictionary, adding NaN values for any missing entries. Each record is timestamped.

        Raises:
            AssertionError: If 'set_csv_columns' has not been called prior to this method.
        """

        assert self.csv_columns is not None, "Must call set_csv_columns first"

        csv_file = self.path / "losses.csv"
        losses.update(
            {
                "date": datetime.datetime.now().isoformat(),
            },
        )
        missing_keys = set(self.csv_columns) - set(losses.keys())

        for key in missing_keys:
            losses[key] = float("nan")

        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.csv_columns,
                quoting=csv.QUOTE_NONNUMERIC,
                extrasaction="raise",
            )
            writer.writerow(losses)
