from pathlib import Path

UNK: float = -99.99


def get_data_dir() -> Path:
    """
    Gets the path to the asset pricing raw_data directory.

    This function returns the path to a directory named 'asset_pricing' located in the user's home directory.
    It is assumed that this directory contains relevant raw_data for asset pricing models.

    Return:
        Path: The path object pointing to the 'asset_pricing' raw_data directory in the user's home directory.
    """

    return Path.home() / "data/asset_pricing"
