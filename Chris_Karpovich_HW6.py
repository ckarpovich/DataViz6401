# Chris Karpovich - Week 6:
# Run with:  streamlit run Chris_Karpovich_HW6.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import streamlit.components.v1 as components
from pyvis.network import Network
from networkx.algorithms.community import greedy_modularity_communities

st.header("College Football Game Network (2000 Season)")
st.subheader("Each node represents a D1-A College Football team while each edge represents a game between two teams that occurred during the 2000 season.")


@st.cache_data
def load_graph():
    return nx.read_gml("football.gml")
G = load_graph()

degrees = dict(G.degree())
communities = list(greedy_modularity_communities(G))

layout = st.sidebar.selectbox("Layout", ["force-directed", "circular"])

color_map = {}
for commid, comm in enumerate(communities):
    for node in comm:
        color_map[node] = commid

PALETTE = ["#2E6E8E", "#d9534f", "#6aa84f", "#b07aa1", "#e6a23c", "#5a6672"]
net = Network(height= "600px", width= "100%", cdn_resources= "in_line", notebook= False, neighborhood_highlight= True)

for n in G.nodes():
    net.add_node(
        n,
        label = n,
        size = 1 + degrees[n] * 1.75,
        color = PALETTE[color_map[n] % len(PALETTE)],
        title = f"Team: {n}\nDegree: {degrees[n]}\nCommunity: {color_map[n]}"
        )

for u, v in G.edges():
    net.add_edge(u, v, color= "#d5d5d5")
    
if layout == "circular":
    pos = nx.circular_layout(G, scale=650)
    for node in net.nodes:
        x, y = pos[node["id"]]
        node["x"], node["y"] = float(x), float(y)
        node["physics"] = False
    net.toggle_physics(False)

st.caption("I chose these encodings to show which teams had more game connections within the D1-A division during the 2000 season. Larger nodes make it easier to identify teams which have more games and therefore more edges. " \
"I chose node color to represent the community that the team is a part of.  This can show us which teams played similiar opponents throughout the season. This encoding demonstrates clusters of" \
" teams with closely connected schedules.")

st.caption("One limitation for this network is that it does not necessarily represent real world categories.  In real life, conferences and strength of schedule are different from the communities represented here." \
" This mainly shows which teams have very similiar opponents on their schedule. This network also suffers slightly from the hairball effect, especially in the circular layout.  " \
" This is why I added the neighborhood highlight feature. It is easier to view when selecting a team and moving the node around, as well as helps make communities even easier to see when a node is selected.")

components.html(net.generate_html(notebook= False), height= 620)
