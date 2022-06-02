import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import altair as alt

@st.experimental_singleton
def get_data(currency="BTC"):
    chart_res = requests.get(
    f"https://asia.deribit.com/api/v2/public/get_tradingview_chart_data?end_timestamp={int(datetime.now().timestamp()*1000)}&instrument_name={currency}-PERPETUAL&resolution=1D&start_timestamp={int((datetime.now()-timedelta(days=1000)).timestamp()*1000)}"
)
    df = pd.DataFrame(chart_res.json()["result"])
    df = df.astype({"ticks": "datetime64[ms]"})
    df.set_index("ticks", inplace=True)

    df["change"] = df["close"].pct_change()
    df["log_close"] = np.log(df["close"])
    df["log_diff"] = df["log_close"].diff()
    df["change_squared"] = df["change"] * df["change"]
    df["rolling30_std"] = df["change"].rolling(30).std()
    df["rolling30_volatility"] = df["rolling30_std"] * np.sqrt(365)
    df["rolling365_std"] = df["change"].rolling(365).std()
    df["rolling365_volatility"] = df["rolling365_std"] * np.sqrt(365)
    df["rolling365_std_ma"] = df["rolling365_std"].rolling(30).mean()
    df["close_365ma"] = df["close"].rolling(365).mean()
    df["close_30ma"] = df["close"].rolling(30).mean()
    return df


st.set_page_config(layout="wide")
option = st.sidebar.selectbox("Currencies:", ["BTC", "ETH"], on_change=get_data, key="currency")
df = get_data(st.session_state.currency)
output_df = pd.DataFrame()
output_df["rolling15_std"] = df["change"].rolling(15).std()
output_df["rolling30_std"] = df["rolling30_std"].copy()
output_df["rolling365_std"] = df["rolling365_std"].copy()
output_df = output_df.dropna(how="all")
c = st.container()
c.header("Rolling std")
c.line_chart(output_df)
base = alt.Chart(df.dropna())
bar = base.mark_bar().encode(
    alt.X("change:Q", bin=alt.Bin(base=20, maxbins=30), type="ordinal"), y="count()"
)
std_col1, std_col2, std_col3 = c.columns(3)
std_col1.metric(
    label="15D",
    value="{value:.4%}".format(value=output_df["rolling15_std"][-1]),
    delta="{delta:.4%}".format(
        delta=output_df["rolling15_std"][-1] - output_df["rolling15_std"][-2]
    ),
)
std_col2.metric(
    label="30D",
    value="{value:.4%}".format(value=output_df["rolling30_std"][-1]),
    delta="{delta:.4%}".format(
        delta=output_df["rolling30_std"][-1] - output_df["rolling30_std"][-2]
    ),
)
std_col3.metric(
    label="365D",
    value="{value:.4%}".format(value=output_df["rolling365_std"][-1]),
    delta="{delta:.4%}".format(
        delta=output_df["rolling365_std"][-1] - output_df["rolling365_std"][-2]
    ),
)
# mean_line = base.mark_rule(color='red').encode(x="mean(change):Q", size=alt.value(3))
c2 = st.container()
c2.header("Histrogram of daily changes")
c2.altair_chart(bar.interactive(), use_container_width=True)
col1, col2, col3, col4 = c2.columns(4)
col1.metric(label="Mean", value="{mean:.4%}".format(mean=df["change"].mean()))
col2.metric(label="Median", value="{median:.4%}".format(median=df["change"].median()))
col3.metric(label="Skew", value="{skew:.4f}".format(skew=df['change'].skew()))
col4.metric(label="Kurtosis", value="{kurtosis:.4f}".format(kurtosis=df['change'].kurt()))