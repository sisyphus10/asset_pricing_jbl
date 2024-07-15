import numpy as np
import pytest

from src.models.util import build_transformer_model


@pytest.fixture
def features_and_mask():
    # seed the PRNG
    np.random.seed(0)
    n_assets: int = 3
    n_time_steps: int = 10
    n_features: int = 4

    features = np.reshape(
        np.arange(n_assets * n_time_steps * n_features),
        (n_time_steps, n_assets, n_features),
    )
    mask = (np.random.rand(n_time_steps, n_assets) > 0.5).astype(float)
    return features, mask


def test_build_transformer_model(features_and_mask):
    features: np.ndarray = features_and_mask[0]
    mask: np.ndarray = features_and_mask[1]
    n_time_steps, n_assets, n_features = features.shape

    model = build_transformer_model(
        output_units=1, num_assets=n_assets, num_features=n_features
    )
    y = model((features, mask))

    assert y.shape == (n_time_steps, n_assets, 1)
