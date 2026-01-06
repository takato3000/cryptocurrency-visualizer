import numpy as np
from numba import jit
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
