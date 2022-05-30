import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import altair as alt



chart_res = requests.get(f"https://asia.deribit.com/api/v2/public/get_tradingview_chart_data?end_timestamp={int(datetime.now().timestamp()*1000)}&instrument_name=BTC-PERPETUAL&resolution=1D&start_timestamp={int((datetime.now()-timedelta(days=1000)).timestamp()*1000)}")
df2 = pd.DataFrame(chart_res.json()['result'])
df2 = df2.astype({'ticks':'datetime64[ms]'})
df2.set_index('ticks', inplace=True)

df2['change'] = df2['close'].pct_change()
df2['log_close'] = np.log(df2['close'])
df2['log_diff'] = df2['log_close'].diff()
df2['change_squared'] = df2['change'] * df2['change']
df2['rolling30_std'] = df2['change'].rolling(30).std()
df2['rolling30_volatility'] = df2['rolling30_std'] * np.sqrt(365)
df2['rolling365_std'] = df2['change'].rolling(365).std()
df2['rolling365_volatility'] = df2['rolling365_std'] * np.sqrt(365)
df2['rolling365_std_ma'] = df2['rolling365_std'].rolling(30).mean()
df2['close_365ma'] = df2['close'].rolling(365).mean()
df2['close_30ma'] = df2['close'].rolling(30).mean()

output_df = pd.DataFrame()
output_df['rolling15_std'] = df2['change'].rolling(15).std()
output_df['rolling30_std'] = df2['rolling30_std'].copy()
output_df['rolling365_std'] = df2['rolling365_std'].copy()
output_df = output_df.dropna(how="all")

st.set_page_config(layout="wide")
options = st.sidebar.selectbox("Currencies:", ["BTC", "ETH", "SOL"] )
c = st.container()
c.header("Rolling std")
c.line_chart(output_df)
base = alt.Chart(df2)
bar = base.mark_bar().encode(alt.X("change:Q", bin=alt.Bin(base=20, maxbins=30), type="ordinal"), y='count()')
std_col1, std_col2, std_col3 = c.columns(3)
std_col1.metric(label="15D", value="{value:.6f}".format(value=output_df['rolling15_std'][-1]), delta="{delta:.6f}".format(delta=output_df['rolling15_std'][-1] - output_df['rolling15_std'][-2]))
std_col2.metric(label="30D", value="{value:.6f}".format(value=output_df['rolling30_std'][-1]), delta="{delta:.6f}".format(delta=output_df['rolling30_std'][-1] - output_df['rolling30_std'][-2]))
std_col3.metric(label="365D", value="{value:.6f}".format(value=output_df['rolling365_std'][-1]), delta="{delta:.6f}".format(delta=output_df['rolling365_std'][-1] - output_df['rolling365_std'][-2]))
# mean_line = base.mark_rule(color='red').encode(x="mean(change):Q", size=alt.value(3))
c2 = st.container()
c2.header("Histrogram of daily changes")
c2.altair_chart(bar.interactive(), use_container_width=True)
col1, col2 = c2.columns(2)
col1.metric(label="Mean", value="{mean:.6f}".format(mean = df2['change'].mean()))
col2.metric(label="Median", value="{median:.6f}".format(median = df2['change'].median()))