from typing import Union
import numpy as np
from pandas import DataFrame
from scipy.stats import norm

class BlackScholesParams:

    # Underlying price (per share): S;
    # Strike price of the option (per share): K;
    # Time to maturity (years): T;
    # Continuously compounding risk-free interest rate: r, 0.03 means 3%;
    # Volatility: sigma; example:15% means 0.15 in sigma;

    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def d1(self):
        return (
            (np.log(self.S / self.K) + (self.r + self.sigma**2 / 2.0) * self.T)
            / self.sigma
            * np.sqrt(self.T)
        )

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        return self.S * norm.cdf(self.d1()) - self.K * np.exp(
            -self.r * self.T
        ) * norm.cdf(self.d2())

    def put_price(self):
        return self.K * np.exp(-self.r * self.T) - self.S + self.call_price()

    def call_delta(self):
        return norm.cdf(self.d1())

    def call_gamma(self):
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def call_theta(self):
        return 0.01 * (
            -(self.S * norm.pdf(self.d1()) * self.sigma) / (2 * np.sqrt(self.T))
            - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())
        )

    def call_vega(self):
        return 0.01 * (self.S * norm.pdf(self.d1()) * np.sqrt(self.T))

    def call_rho(self):
        return 0.01 * (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2()))

    # Put part below

    def put_delta(self):
        return -norm.cdf(-self.d1())

    def put_gamma(self):
        return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))

    def put_vega(self):
        return 0.01 * (self.S * norm.pdf(self.d1()) * np.sqrt(self.T))

    def put_theta(self):
        return 0.01 * (
            -(self.S * norm.pdf(self.d1()) * self.sigma) / (2 * np.sqrt(self.T))
            + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2())
        )

    def put_rho(self):
        return 0.01 * (
            -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2())
        )


def calculate(
    S: Union[int, float, np.ndarray],
    K: Union[int, float, np.ndarray],
    T: Union[float, np.ndarray],
    r: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
) -> DataFrame:
    if isinstance(S, np.ndarray):
        result = calculate_array(S, K, T, r, sigma)
    else:
        result = calculate_single_point(S, K, T, r, sigma)
    return result


def calculate_array(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, r: np.ndarray, sigma: np.ndarray
) -> DataFrame:
    params = BlackScholesParams(S, K, T, r, sigma)
    # columns = ["strike", "call_price", "call_delta", "put_price", "put_delta", "gamma", "theta", "vega"]
    result = DataFrame()
    result["strike"] = K
    result["call_price"] = params.call_price()
    result["call_delta"] = params.call_delta()
    result["put_price"] = params.put_price()
    result["put_delta"] = params.put_delta()
    result["gamma"] = params.call_gamma()
    result["theta"] = params.call_theta()
    result["vega"] = params.call_vega()
    return result


def calculate_single_point(
    S: Union[int, float],
    K: Union[int, float],
    T: Union[int, float],
    r: Union[int, float],
    sigma: Union[int, float],
) -> DataFrame:
    result = DataFrame(
        data=np.zeros((1, 8)),
        columns=[
            "strike",
            "call_price",
            "call_delta",
            "put_price",
            "put_delta",
            "gamma",
            "theta",
            "vega",
        ],
    )
    result.loc[0, "strike"] = K
    params = BlackScholesParams(S, K, T, r, sigma)
    result.loc[0, "call_price"] = params.call_price()
    result.loc[0, "call_delta"] = params.call_delta()
    result.loc[0, "put_price"] = params.put_price()
    result.loc[0, "put_delta"] = params.put_delta()
    result.loc[0, "gamma"] = params.call_gamma()
    result.loc[0, "theta"] = params.call_theta()
    result.loc[0, "vega"] = params.call_vega()
    return result