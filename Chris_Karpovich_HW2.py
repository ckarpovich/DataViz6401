# Chris Karpovich - Week 2:
# Run with:  streamlit run Chris_Karpovich_HW2.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("mlbstats.csv")
df["HR_class"] = pd.cut(df["HR"], bins= 3, labels = ["Few", "Medium", 'Many'])

st.title("How does Team Home Run Total Impact Team Walks and Strikeouts") 
st.caption("Analysts suggest that the more home runs a team hits, the more likely they are to strike out. " \
"Let's see if we can put that to the test. Since MLB has 30 teams and discerning 30 shapes would be hard, I divided it into 3 groups.")

size, color, shape = st.columns(3)

with size:
    st.subheader("HR as Size")
    fig, ax = plt.subplots()
    sns.scatterplot(data= df, x= "BB", y= "SO", size = "HR", ax = ax)
    st.pyplot(fig)
    st.caption("Larger points generally would suggest more home runs at first glance.  Similarly, larger points stand out and easily identify" \
    "teams with large home run totals. ")

with color:
    st.subheader("HR as Color")
    fig, ax = plt.subplots()
    sns.scatterplot(data= df, x= "BB", y= "SO", hue= "HR", ax = ax)
    st.pyplot(fig)
    st.caption("Color allows for quick pattern recognition and grouping of similar values. However this plot has colors that are not the most discernable. Color is also less precise than channels like size or position.")

with shape:
    st.subheader("HR as Shape")
    fig, ax = plt.subplots()
    sns.scatterplot(data= df, x= "BB", y= "SO", style = "HR_class", ax = ax)
    st.pyplot(fig)
    st.caption("Shape is very good for determining which bucket the team's HR total falls in, but it is not great for specific values.  That is why I decided to cut the HR totals into 3 buckets, however some of the numerical value is lost in this chart because we had to split it up into 3 buckets.")