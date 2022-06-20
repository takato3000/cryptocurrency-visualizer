import numpy as np
import pytest
import pandas as pd

from black_scholes_functions import *
from black_scholes import *

@pytest.fixture
def data():
    return dict({
        "S": np.array([300, 300]),
        "K": np.array([250, 350]),
        "T": np.array([1.0, 1.0]),
        "r": np.array([0.03, 0.03]),
        "sigma": np.array([0.15, 0.15]),
    })

def test_d1(data):
    assert np.allclose(d1(**data), [1.490, -0.753], atol=1e-3)


def test_call_price(data):
    assert np.allclose(call_price(**data), [58.820, 5.471], atol=1e-3)


def test_call_delta(data):
    ans = call_delta(**data)
    assert np.allclose(ans, [0.932, 0.226], atol=1e-3)

def test_bs_class(data):
    bs = BlackScholesParams(**data)
    assert bs.__dict__ == data

def test_bs_in_dataframe(data):
    result = calculate_array(**data)
    assert np.allclose(result["call_delta"].to_numpy(), [0.932, 0.226], atol=1e-3)
    assert np.allclose(result["call_price"].to_numpy(), [58.820, 5.471], atol=1e-3)

