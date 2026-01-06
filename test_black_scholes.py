import numpy as np
import unittest
import pytest

from black_scholes_functions import (
    BlackScholesParams as BlackScholesParams
)
from black_scholes import (
    calculate_array as calculate_array,
    calculate_single_point as calculate_single_point,
)


class TestBlackScholesFunctions(unittest.TestCase):
    def setUp(self):
        self.BSP = BlackScholesParams(np.array([300, 300]), np.array([250, 350]), np.array([1.0, 1.0]), np.array([0.03, 0.03]), np.array([0.15, 0.15]))
        self.data = dict(
            {
                "S": np.array([300, 300]),
                "K": np.array([250, 350]),
                "T": np.array([1.0, 1.0]),
                "r": np.array([0.03, 0.03]),
                "sigma": np.array([0.15, 0.15]),
            }
        )
        return super().setUp()
    
    def test_d1(self):
        assert np.allclose(self.BSP.d1(), [1.490, -0.753], atol=1e-3)
    
    def test_call_price(self):
        assert np.allclose(self.BSP.call_price(), [58.820, 5.471], atol=1e-3)
    
    def test_call_delta(self):
        assert np.allclose(self.BSP.call_delta(), [0.932, 0.226], atol=1e-3)
    
    def test_bs_in_dataframe(self):
        result = calculate_array(**self.data)
        assert np.allclose(result["call_delta"].to_numpy(), [0.932, 0.226], atol=1e-3)
        assert np.allclose(result["call_price"].to_numpy(), [58.820, 5.471], atol=1e-3)


def suite():
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBlackScholesFunctions))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())