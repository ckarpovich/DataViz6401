# Chris Karpovich - Week 4:
# Run with:  streamlit run Chris_Karpovich_HW4.py
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
"NFL team, the Baltimore Ravens")

df = pd.read_csv("ravensgoogletrends.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
monthly = df["interest"]
rule = st.selectbox ("Select a time resolution for first chart (Monthly, Quarterly, Yearly)", ["MS", "QS", "YS"])
shown = monthly.resample(rule).mean()


#Line Chart
st.subheader("Baltimore Ravens Search Interest Since 2004")
fig, ax = plt.subplots()
shown.plot(ax = ax, figsize = (9,4))
ax.set_ylabel('Google Trends Interest')
ax.set_title("Ravens Search Interest Since 2004")
st.caption("Temporal-Honesty: I kept the monthly data from my export so that the seasonal patterns would be easily visible.  For the first line graph," \
"I allowed the user of the app to change resolution to quarterly or yearly to see patterns over different increments of time.  Seasonal decomposition still keeps those monthly increments." \
"It is important to note that when using the yearly or quarterly options, 2026 is not fully complete.  This could influence one's interpretation of the charts if not properly noted. ")
st.pyplot(fig)


#12 Month Rolling Mean
st.subheader("12 Month Rolling Mean")
fig, ax = plt.subplots()
monthly.plot(alpha=0.4, label="monthly", figsize=(9, 4))
monthly.rolling(12).mean().plot(ax=ax, label="12-month rolling mean", linewidth=2)
ax.set_ylabel('Google Trends Interest')
ax.set_title("Ravens Search Interest Since 2004")
ax.legend()
st.caption("")
st.pyplot(fig)

#Uncertainty Chart
st.subheader("Baltimore Ravens Search Interest with Uncertainty")
roll = monthly.rolling(12)
mean, std = roll.mean(), roll.std()

fig, ax = plt.subplots()
mean.plot(figsize=(9, 4), label="12-month mean")
ax.fill_between(mean.index, mean - 2 * std, mean + 2 * std, alpha=0.2, label="±2σ band")
ax.set_ylabel('Google Trends Interest')
ax.set_title("Ravens Search Interest w/ Uncertainty")
ax.legend()
st.caption("")
st.pyplot(fig)

#Seasonal Decomposition
result = seasonal_decompose(monthly, period=12)
fig = result.plot()
plt.tight_layout()
st.caption("")
st.pyplot(fig)