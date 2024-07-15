import logging
from typing import Literal

import numpy as np

from src import get_data_dir


def load_equity_data(name: Literal["train", "valid", "test"] = "train"):
    """
    Loads asset raw_data from a specified dataset (training, validation, or test).

    Args:
        name (Literal["train", "valid", "test"], optional): The name of the dataset to load.
                                                            Defaults to "train".

    This function loads asset raw_data from a NumPy compressed file (.npz) corresponding to the specified dataset.
    The raw_data should be located in the 'char' subdirectory within the 'asset_pricing' raw_data directory.

    Raises:
        AssertionError: If the specified dataset name is not one of 'train', 'valid', or 'test'.

    Return:
        ndarray: The loaded raw_data from the specified dataset.
    """

    assert name in [
        "train",
        "valid",
        "test",
    ], "name must be one of 'train', 'valid', 'test'"

    datasets = get_data_dir() / "datasets"
    char = datasets / "char"
    char_train = char / f"Char_{name}.npz"

    # char_train = char / "Char_valid.npz"
    logging.info(f"Loading raw_data from '{char_train}'")
    ar = np.load(char_train)

    data = ar["data"]
    feature_names = ar.f.variable[1:]
    return data, feature_names


def load_macro_data(name: Literal["train", "valid", "test"] = "train"):
    """
    Loads macroeconomic raw_data from a specified dataset (training, validation, or test).

    Args:
        name (Literal["train", "valid", "test"], optional): The name of the dataset to load.
                                                            Defaults to "train".

    This function loads macroeconomic raw_data from a NumPy compressed file (.npz) corresponding to the specified dataset.
    The raw_data should be located in the 'macro' subdirectory within the 'asset_pricing' raw_data directory.

    Raises:
        AssertionError: If the specified dataset name is not one of 'train', 'valid', or 'test'.

    Return:
        ndarray: The loaded macroeconomic raw_data from the specified dataset.
    """
    assert name in [
        "train",
        "valid",
        "test",
    ], "name must be one of 'train', 'valid', 'test'"

    datasets = get_data_dir() / "datasets"
    macro = datasets / "macro"
    macro_train = macro / f"macro_{name}.npz"

    logging.info(f"Loading raw_data from '{macro_train}'")
    ar = np.load(macro_train)

    data = ar["data"]
    return data
