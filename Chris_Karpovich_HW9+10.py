# Chris Karpovich - Week 9-10:
# Run with:  streamlit run Chris_Karpovich_HW9+10.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
import geopandas as gpd
import streamlit.components.v1 as components
from shapely.geometry import Point


st.set_page_config(page_title="Supercharger Analysis", layout="wide")
st.title("Supercharger Locations in the USA")
st.caption("Analyzing distribution of Tesla Supercharger locations across the United States")

tab, tab2 = st.tabs(["Supercharger Map", "County Population Analysis"])

df = pd.read_csv("superchargers_us.csv")
pop = pd.read_csv("countypop25.csv")
c = gpd.read_file("tl_2026_us_county/tl_2026_us_county.shp")

#Keep only US States (Not outlying terriroties)
c = c[c["STATEFP"].astype(int) <= 56]

mincharge = st.sidebar.slider(
     "Minimum Charging Power in KW",
     min_value = int(df["kw"].min()),
     max_value = int(df["kw"].max()),
     value = int(df["kw"].min()),
     step = 10
)

df = df[df["kw"] >= mincharge]
st.sidebar.write("Locations remaining:", len(df))

c = c.to_crs(epsg = 5070)

#Keep the 0's at the start of each FIPS Code
pop["FIPS Code"] = pop["FIPS Code"].astype(str).str.zfill(5)

pts = gpd.GeoDataFrame(
    df,
    geometry=[Point(coord) for coord in zip(df["lon"], df["lat"])],
    crs="EPSG:4326")

pts = pts.to_crs(epsg = 5070)

joined = gpd.sjoin(pts, c[["GEOID", "geometry"]], predicate="within")

unmatched = len(pts) - len(joined)

counts = joined.groupby("GEOID").size().reset_index(name = "Superchargers")

county = c.merge(counts, on="GEOID", how = 'left')
county["Superchargers"] = county["Superchargers"].fillna(0)
county = county.merge(pop, left_on = "GEOID", right_on = "FIPS Code", how = "left")

county["Rate"] = county["Superchargers"] / county["Total Population"] * 100000

joined = joined.merge(county[["GEOID", "Rate"]], on="GEOID", how="left")
county["geometry"] = county.geometry.simplify(1000)
county = county.to_crs(epsg = 4326)

with tab:
    m = folium.Map(location = [df["lat"].mean(), df["lon"].mean()], zoom_start = 4)

    for i, row in df.iterrows():
        folium.CircleMarker(location = [row["lat"], row["lon"]],
                      popup = row["name"],
                      radius = 3.5,
                      color = "darkblue",
                      fill = True,
                      fill_color = "darkblue"
        ).add_to(m)

    st.caption("Each blue circle represents a Tesla Supercharger location in the United States.  The US outlying islands and territories (like Puerto Rico) are excluded from this analysis. The slider filters the locations by their minimum charging power in kilowatts.  Only locations that have the same or more than the selected amount of power are shown.")

    components.html(m._repr_html_(), height = 600)

with tab2:
    m2 = folium.Map(location = [df["lat"].mean(), df["lon"].mean()], zoom_start = 4)

    st.caption(f"Supercharger Locations not matched to a county: {unmatched}")

    folium.Choropleth(
        geo_data = county,
        data = county,
        columns = ["GEOID", "Rate"],
        key_on = "feature.properties.GEOID",
        bins = 6,
        fill_color= "YlOrRd",
        fill_opacity = 0.8,
        line_opacity = 0.3,
        legend_name = "Supercharger Locations per 100,000 Residents"
    ).add_to(m2)

    for i, row in joined.iterrows():
            folium.CircleMarker(location = [row["lat"], row["lon"]],
                                radius = 3.5,
                                popup=f"{row['name']}<br>Rate per 100,000 residents: {row['Rate']:.2f}",
                                color = "darkblue",
                                fill = True,
                                fill_color = "darkblue"
            ).add_to(m2)

    st.caption("The shading of each county represents Tesla Supercharger locations per 100,000 residents.  The rates are calculated by dividing the number of locations in a specific county by its population and multiplying by 100,000.")
    st.caption("Population data was joined to county boundaries using GEOID and FIPS codes. The boundaries were reprojected to EPSG:5070 for spatial analysis and converted to EPSG: 4326 for display.  The Choropleth uses 6 classes (equal interval) and a yellow to orange to red color scale.")
    st.caption("Each blue circle represents a charging location.  The slider on the sidebar filters the locations by their minimum charging power.  Rates are calculated again based on the minimum power chosen by the user.  The 3 unmatched locations come from US outlying islands, which were removed from the dataset.")
    components.html(m2._repr_html_(), height = 600)

    st.caption("Most counties have no Tesla Supercharger locations.  Counties with higher rates do not necessarily have more charging stations, since small populations can produce higher rates with just one or two locations.  This chart highlights counties with those high rates, and mostly shows counties with low populations compared to others with one or two charger locations.")

