# Chris Karpovich - Week 11-12:
# Run with:  streamlit run Chris_Karpovich_HW11+12.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

old = pd.read_csv("Batting.csv")
old = old[old["AB"] > 0]
old["avg"] = old["H"] / old["AB"]

old = old.sort_values(["playerID", "yearID"])

old["newavg"] = old.groupby("playerID")["avg"].shift(-1)
old["above"] = (old["newavg"] >= .200).astype(int)
old = old.dropna(subset=["avg", "HR", "H", "SO", "BB", "RBI", "newavg"])

X = old[["avg", "HR", "H", "SO", "BB", "RBI"]]
y = old["above"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 25)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write("Accuracy: ", acc)

st.title("MLB Player Batting Average Prediction")
st.caption("Predicting whether a player will hit above or below a .200 batting average next season (Mendoza Line)")

st.caption("""
Example Player IDs:
- Shohei Ohtani: 660271
- Aaron Judge: 592450
- Freddie Freeman: 518692
- Kyle Schwarber: 656941
- Cal Raleigh: 663728
- Colson Montgomery: 695657
"""
)

BASE = "http://statsapi.mlb.com/api/v1"

playerid = st.text_input("Enter an MLB Player ID", "660271")

@st.cache_data(ttl=300)
def fetch(base, playerid):
    params = {"stats":"season",
              "group": "hitting",
              "season": 2026
        }
    r = requests.get(f"{base}/people/{playerid}/stats", params=params, timeout=5)
    r.raise_for_status()
    return r.json()

if st.button("Fetch"):
    try:
        payload = fetch(BASE, playerid)

        namedata = requests.get(f"{BASE}/people/{playerid}", timeout = 5)
        namedata.raise_for_status()
        name = namedata.json()["people"][0]["fullName"]
        st.subheader("Player")
        st.write(name)

        stats = payload["stats"][0]["splits"]

        if len(stats) == 0:
            st.error("Error: No hitting statistics available")
        else:
            df = pd.DataFrame([item["stat"] for item in stats])

            df = df[["avg", "homeRuns", "hits", "strikeOuts", "baseOnBalls", "rbi"]]
            df = df.astype(float)

            stats2 = pd.DataFrame([{
                "avg": df["avg"].iloc[0],
                "HR": df["homeRuns"].iloc[0],
                "H": df["hits"].iloc[0],
                "SO": df["strikeOuts"].iloc[0],
                "BB": df["baseOnBalls"].iloc[0],
                "RBI": df["rbi"].iloc[0]
            }])

            pred = model.predict(stats2)

        st.subheader("2026 Batting Average")
        st.write(df["avg"].iloc[0])

        if pred[0] == 1:
            st.success("Player is projected to hit above the Mendoza Line in 2027")
        else:
            st.write("Player is projected to hit below the Mendoza Line in 2027")

    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        st.caption("Is the server running? Is the URL right?")
