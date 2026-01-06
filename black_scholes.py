from typing import Union

import numpy as np
import streamlit as st
from pandas import DataFrame
import black_scholes_functions as bs


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
    params = bs.BlackScholesParams(S, K, T, r, sigma)
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
    params = bs.BlackScholesParams(S, K, T, r, sigma)
    result.loc[0, "call_price"] = params.call_price()
    result.loc[0, "call_delta"] = params.call_delta()
    result.loc[0, "put_price"] = params.put_price()
    result.loc[0, "put_delta"] = params.put_delta()
    result.loc[0, "gamma"] = params.call_gamma()
    result.loc[0, "theta"] = params.call_theta()
    result.loc[0, "vega"] = params.call_vega()
    return result

if __name__ == "__main__":
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
            st.write(f"Underlying Price: {S}")
            st.dataframe(result)
