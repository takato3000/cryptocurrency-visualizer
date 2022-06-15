from datetime import date, datetime, timedelta
from math import exp, log, pi, sqrt

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from numba import jit
from pandas import DataFrame
from scipy.stats import norm
import black_scholes_functions as bs

def calculate(S, K, T, r, sigma):
    result = DataFrame(data=np.zeros((1, 7)), columns=["call_price", "call_delta", "put_price", "put_delta", "gamma", "theta", "vega"])
    result["call_price"][0] = bs.call_price(S, K, T, r, sigma)
    result["call_delta"][0] = bs.call_delta(S, K, T, r, sigma)
    result["put_price"][0] = bs.put_price(S, K, T, r, sigma)
    result["put_delta"][0] = bs.put_delta(S, K, T, r, sigma)
    result["gamma"][0] = bs.call_gamma(S, K, T, r, sigma)
    result["theta"][0] = bs.call_theta(S, K, T, r, sigma)
    result["vega"][0] = bs.call_vega(S, K, T, r, sigma)
    return result.copy()

st.set_page_config(page_title="Black Scholes Calculator", layout="wide")
with st.form(key="parameters"):
    S = st.number_input("Underlying Price(SP)", 0)
    K = st.number_input("Strike Price(ST)")
    T = st.number_input("Time to Expiration in Years(t)")
    sigma = st.number_input("Volatility(v)")
    r = st.number_input("Risk-Free Interest Rate(r)")
    submitted = st.form_submit_button("Calculate")
    if submitted:
        result = calculate(S, K, T, r, sigma)
        st.dataframe(result)