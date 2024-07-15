from __future__ import annotations

import numpy as np
import pytest

UNK: float = -99.99


@pytest.fixture(scope="module")
def data_with_unknowns():
    n_assets: int = 10
    n_months: int = 12
    n_features: int = 2

    np.random.seed(0)
    ret = np.random.randn(n_months, n_assets) * 0.02
    ret[0, 0] = UNK

    features = np.random.randn(n_months, n_assets, n_features) * 0.02

    return np.concatenate([ret[:, :, None], features], axis=-1)


def test_data(data_with_unknowns):
    assert UNK in data_with_unknowns
    assert np.sum(UNK == data_with_unknowns) == 1
    assert data_with_unknowns.shape == (12, 10, 3)
