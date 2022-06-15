import numpy as np
import pytest

from black_scholes_functions import *

test_data = {
    "S": np.array([300, 300]),
    "K": np.array([250, 350]),
    "T": np.array([1.0, 1.0]),
    "r": np.array([0.03, 0.03]),
    "sigma": np.array([0.15, 0.15]),
}

def test_d1():
    data = test_data.copy()
    assert np.allclose(d1(**data), [1.490, -0.753], atol=1e-3)


def test_call_price():
    data = test_data.copy()
    assert np.allclose(call_price(**data), [58.820, 5.471], atol=1e-3)


def test_call_delta():
    data = test_data.copy()
    ans = call_delta(**data)
    assert np.allclose(ans, [0.932, 0.226], atol=1e-3)
