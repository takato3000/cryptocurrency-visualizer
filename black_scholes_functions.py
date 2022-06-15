import numpy as np
from numba import jit
from scipy.stats import norm

# Underlying price (per share): S;
# Strike price of the option (per share): K;
# Time to maturity (years): T;
# Continuously compounding risk-free interest rate: r;
# Volatility: sigma; example:15% means 0.15 in sigma;


class BlackScholes:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma


@jit(nopython=True)
def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + sigma ** 2 / 2.0) * T) / sigma * np.sqrt(T)


@jit(nopython=True)
def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    return S * norm.cdf(d1(S, K, T, r, sigma)) - K * np.exp(-r * T) * norm.cdf(
        d2(S, K, T, r, sigma)
    )


def put_price(S, K, T, r, sigma):
    return K * np.exp(-r * T) - S + call_price(S, K, T, r, sigma)


def call_delta(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma))


def call_gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))


def call_theta(S, K, T, r, sigma):
    return 0.01 * (
        -(S * norm.pdf(d1(S, K, T, r, sigma)) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))
    )


def call_vega(S, K, T, r, sigma):
    return 0.01 * (S * norm.pdf(d1(S, K, T, r, sigma)) * np.sqrt(T))


def call_rho(S, K, T, r, sigma):
    return 0.01 * (K * T * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma)))


# Put part below


def put_delta(S, K, T, r, sigma):
    return -norm.cdf(-d1(S, K, T, r, sigma))


def put_gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))


def put_vega(S, K, T, r, sigma):
    return 0.01 * (S * norm.pdf(d1(S, K, T, r, sigma)) * np.sqrt(T))


def put_theta(S, K, T, r, sigma):
    return 0.01 * (
        -(S * norm.pdf(d1(S, K, T, r, sigma)) * sigma) / (2 * np.sqrt(T))
        + r * K * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma))
    )


def put_rho(S, K, T, r, sigma):
    return 0.01 * (-K * T * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma)))
