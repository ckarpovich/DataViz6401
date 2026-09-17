# Chris Karpovich - Week 3:
# Run with:  streamlit run Chris_Karpovich_HW3.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv("mlbstats.csv")
df["HR_class"] = pd.cut(df["HR"], bins= 3, labels = ["Few", "Medium", 'Many'])
num = df[["R", "H", "RBI","2B", "3B", "HR", "BB", "SO"]]

left, right = st.columns(2)

#Heatmap
with left:
    st.subheader("Correlation among MLB Teams' Offensive Stats")
    fig, ax = plt.subplots()
    sns.heatmap(num.corr(), cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.6})
    st.pyplot(fig)
    st.caption("")

#PCA
X = StandardScaler().fit_transform(num)
pcs = PCA(n_components=2).fit_transform(X)


with right:
    st.subheader("MLB Offensive Stats Explorer")
    plot_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
    plot_df["HR_class"] = df["HR_class"].astype(str)
    color_col = st.selectbox("Color points by", ["HR_class"])

    st.scatter_chart(plot_df, x="PC1", y="PC2", color=color_col)
    st.caption("Looking at the scatterplot, MLB teams (represented by each dot) that belong to similar home run categories tend to have similar offensive stats." \
    "The dark blue (many HR group) tends to be towards the upper right quadrant.  The orange (medium HR group) tend to stay in the middle of both axes towards the (0,0) point.  Light blue" \
    "(few HR group) tends to be in the lower left hand quadrant. This shows that the teams that fall into each HR category/class tend to have different totals for the other offensive stats" \
    "as well.  PC1 and PC2 simply represent the main pattern of variation between each MLB team, where PC2 represents the second most impactful difference that isn't already a part of PC1.")
