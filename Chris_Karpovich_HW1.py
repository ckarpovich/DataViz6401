# Chris Karpovich - Week 1: your first Streamlit app
# Run with:  streamlit run Chris_Karpovich_HW1.py
import streamlit as st
import pandas as pd

df = pd.read_csv("mlbstats.csv")
df = df[["Tm", "HR"]]
df = df.sort_values("HR")

st.title("MLB Home Runs by Team") 
st.caption("Home Run Distribution by MLB Team in 2026 Season (So Far)")

st.dataframe(df)
st.bar_chart(df, x="Tm", y="HR")
