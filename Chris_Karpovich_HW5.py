# Chris Karpovich - Week 5:
# Run with:  streamlit run Chris_Karpovich_HW5.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose

st.header("Baltimore Ravens Google Search Trends")
st.subheader("Trend, Seasonality, and Uncertainty from data taken from 2004-2026")
st.caption("The data is provided by Google Trends. The data is normalized on a 0-100 scale with 100 representing the peak popularity in Google interest for the" \
" NFL team, the Baltimore Ravens")

@st.cache_data
def load_data():
    df = pd.read_csv("ravensgoogletrends.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df["interest"]

monthly = load_data()

#Controls
rule = st.sidebar.selectbox ("Select a time resolution for first chart (Monthly, Quarterly, Yearly)", ["MS", "QS", "YS"])

months = st.sidebar.slider("Rolling window (months)", min_value = 1, max_value=36, value = 12)
shown = monthly.resample(rule).mean()

#Columns
col1, col2= st.columns(2)

col1.metric("Time Resolution", rule)
col2.metric("Rolling Window ",f"{months} months")

#Line Chart
st.subheader("Baltimore Ravens Search Interest Since 2004")
fig, ax = plt.subplots()
shown.plot(ax = ax, figsize = (9,4))
ax.set_ylabel('Google Trends Interest')
ax.set_title("Ravens Search Interest Since 2004")
st.caption("Temporal-Honesty: I kept the monthly data from my export so that the seasonal patterns would be easily visible.  For the first line graph," \
"I allowed the user of the app to change resolution to quarterly or yearly to see patterns over different increments of time.  Seasonal decomposition still keeps those monthly increments." \
" It is important to note that when using the yearly or quarterly options, 2026 is not fully complete.  This could influence one's interpretation of the charts if not properly noted. ")
st.pyplot(fig)
plt.close(fig)


#12 Month Rolling Mean
st.subheader("Rolling Mean")
fig, ax = plt.subplots()
monthly.plot(alpha=0.4, label="monthly", figsize=(9, 4))
monthly.rolling(months).mean().plot(ax=ax, label= f"{months} month rolling mean", linewidth=2)
ax.set_ylabel('Google Trends Interest')
ax.set_title("Ravens Search Interest Since 2004")
ax.legend()
st.caption("")
st.pyplot(fig)
plt.close(fig)

#Uncertainty Chart
st.subheader("Baltimore Ravens Search Interest with Uncertainty")
roll = monthly.rolling(months)
mean, std = roll.mean(), roll.std()

fig, ax = plt.subplots()
mean.plot(figsize=(9, 4), label=f"{months} month rolling mean")
ax.fill_between(mean.index, mean - 2 * std, mean + 2 * std, alpha=0.2, label="±2σ band")
ax.set_ylabel('Google Trends Interest')
ax.set_title("Ravens Search Interest w/ Uncertainty")
ax.legend()
st.caption("")
st.pyplot(fig)
plt.close(fig)

#Seasonal Decomposition
result = seasonal_decompose(monthly, period=12)
fig = result.plot()
plt.tight_layout()
st.caption("")
st.pyplot(fig)
plt.close(fig)