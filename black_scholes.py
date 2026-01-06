from black_scholes_functions import calculate
import streamlit as st



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
