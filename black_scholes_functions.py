from datetime import date, datetime, timedelta
from math import exp, log, pi, sqrt

import numpy as np
import pandas as pd
from numba import jit
from pandas import DataFrame
from scipy.stats import norm

# Underlying price (per share): S;
# Strike price of the option (per share): K;
# Time to maturity (years): T;
# Continuously compounding risk-free interest rate: r;
# Volatility: sigma; example:15% means 0.15 in sigma;


@jit(nopython=True)
def d1(S, K, T, r, sigma):
    return (log(S / K) + (r + sigma ** 2 / 2.0) * T) / sigma * sqrt(T)


@jit(nopython=True)
def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * sqrt(T)


def call_price(S, K, T, r, sigma):
    return S * norm.cdf(d1(S, K, T, r, sigma)) - K * exp(-r * T) * norm.cdf(
        d2(S, K, T, r, sigma)
    )


def put_price(S, K, T, r, sigma):
    return K * exp(-r * T) - S + call_price(S, K, T, r, sigma)


def call_delta(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma))


def call_gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * sqrt(T))


def call_theta(S, K, T, r, sigma):
    return 0.01 * (
        -(S * norm.pdf(d1(S, K, T, r, sigma)) * sigma) / (2 * sqrt(T))
        - r * K * exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))
    )


def call_vega(S, K, T, r, sigma):
    return 0.01 * (S * norm.pdf(d1(S, K, T, r, sigma)) * sqrt(T))


def call_rho(S, K, T, r, sigma):
    return 0.01 * (K * T * exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma)))


# Put part below


def put_delta(S, K, T, r, sigma):
    return -norm.cdf(-d1(S, K, T, r, sigma))


def put_gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * sqrt(T))


def put_vega(S, K, T, r, sigma):
    return 0.01 * (S * norm.pdf(d1(S, K, T, r, sigma)) * sqrt(T))


def put_theta(S, K, T, r, sigma):
    return 0.01 * (
        -(S * norm.pdf(d1(S, K, T, r, sigma)) * sigma) / (2 * sqrt(T))
        + r * K * exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma))
    )


def put_rho(S, K, T, r, sigma):
    return 0.01 * (-K * T * exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma)))
