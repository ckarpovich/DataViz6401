# Chris Karpovich - Midterm Project:
# Run with:  streamlit run Chris_Karpovich_Midterm.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv("WV_Weather.csv", low_memory = False)
df["El_class"] = pd.cut(df["ELEVATION"], bins= 3, labels = ["Low", "Medium", 'High'])

plt.style.use("dark_background")
#print(df.head())
#print(df.describe())

st.title("West Virginia Weather Exploration")

ov, map, time, text = st.tabs(["Overview", "Map Exploration", "Weather Over Time", "Weather Type Text Analysis"])

with ov:
    st.header("Overview")
    st.subheader("Research Question")
    st.caption("How do elevation and geography relate to daily weather patterns across West Virginia in 2025?")

    st.subheader("Background")
    st.caption("West Virginia is a state with varied terrain, including prominent mountains as well as low elevation valleys.  This project explores how weather patterns can differ across multiple locations within the same state.")

    st.subheader("Data Overview")
    st.caption("The dataset used in this exploration contains daily weather observations collected from many weather stations across West Virginia in 2025.  These stations are part of the National Oceanic and Atmospheric Administration (NOAA).  Climate records include weather reports from airport stations, volunteer/observer stations, government agencies, and more.  The analysis focuses mostly on maximum temperature, minimum temperature, precipitation, and snowfall.")

    st.subheader("Limitations to Consider")
    st.caption("This data was all recorded in 2025, so it does not reflect long term weather patterns.  These visualizations are meant to show patterns and relationships, not say that elevation or geography directly causes these differences in weather.")
#Map
with map:

    #Elevation Map
    st.header("Geography Explorer")
    st.subheader("Station Elevation (meters)")
    st.caption("Exploring locations and elevations across weather stations of West Virginia.  Each circle represents a weather station from the dataset.  The station ID, city name, and elevation (in meters) are presented in a tooltip when hovering over a station." \
    " Some stations indicate a cardinal direction and a distance in miles from the actual city.  For example, Middlebourne 3.2 Ese represents a station 3.2 miles east and southeast from Middlebourne, WV.  The elevation is colored with lower elevations being light yellow, medium elevations being orange, and high elevations being a dark red.")

    mapdata = df.dropna(subset=["LATITUDE", "LONGITUDE", "ELEVATION"]).copy()
    mapdata = mapdata.drop_duplicates(subset = ["STATION"])
    emin = mapdata["ELEVATION"].min()
    emax = mapdata["ELEVATION"].max()

    m = folium.Map(location=[38.6, -80.6], zoom_start=7, tiles=None, scrollWheelZoom=False)
    folium.TileLayer("OpenStreetMap", control=False).add_to(m)

    for i, row in mapdata.iterrows():
        elev = (row["ELEVATION"] - emin) / (emax - emin)

        ecolor = to_hex(matplotlib.colormaps["YlOrRd"](elev))

        folium.CircleMarker(location = [row["LATITUDE"], row["LONGITUDE"]], 
                            radius = 5, color = ecolor, 
                            fill =True, 
                            fill_color = ecolor, 
                            fill_opacity = 1,
                            tooltip =f"Station: {row['STATION']} | City: {row['CITY']} | Elevation: {row['ELEVATION']} Meters above Sea Level").add_to(m)

    
    components.html(m._repr_html_(), height=550)

    st.caption("After looking at the elevation map, let's take a look at how the elevation of a weather station can relate to the temperatures recorded.")

    #Avg Monthly Max
    st.subheader("Monthly Average Maximum Temperature (Fahrenheit)")
    st.caption("Again, each circle represents a weather station from the dataset. Light yellow indicates lower temperatures, orange indicates medium temperatures, and dark red indicates higher temperatures.  The colors are scaled to the selected month's average maximum daily temperature.")

    maxmonth = st.slider(
        "Select Month for Average Max Temp",
        min_value = 1,
        max_value = 12,
        value = 1
    )

    monthdata = df[df["MONTH"]== maxmonth].copy()
    monthdata = monthdata.dropna(subset=["TMAX", "LATITUDE", "LONGITUDE"])

    tempdata = monthdata.groupby(["STATION", "CITY", "LATITUDE", "LONGITUDE", "ELEVATION"])["TMAX"].mean().reset_index()

    mintemp1 = tempdata["TMAX"].min()
    maxtemp1 = tempdata["TMAX"].max()

    m2 = folium.Map(location=[38.6, -80.6], zoom_start=7, tiles=None, scrollWheelZoom=False)
    folium.TileLayer("OpenStreetMap", control=False).add_to(m2)

    for i, row in tempdata.iterrows():

            tempvalue = (row["TMAX"] - mintemp1) / (maxtemp1 - mintemp1)

            tcolor = to_hex(matplotlib.colormaps["YlOrRd"](tempvalue))

            folium.CircleMarker(location = [row["LATITUDE"], row["LONGITUDE"]], 
                                radius = 5, color = tcolor, 
                                fill =True, 
                                fill_color = tcolor, 
                                fill_opacity = 1,
                                tooltip =f"City: {row['CITY']} | Avg. Max Temp.: {row['TMAX']:.1f} degrees F | Elevation: {row['ELEVATION']} M").add_to(m2)

    components.html(m2._repr_html_(), height=550)

#Avg Monthly Minimums
    st.subheader("Monthly Average Minimum Temperature (Fahrenheit)")
    st.caption("Here, the colors are scaled to the selected month's average minimum daily temperature. Light yellow indicates lower temperatures, orange indicates medium temperatures, and dark red indicates higher temperatures.")

    minmonth = st.slider(
            "Select Month for Average Min Temp",
            min_value = 1,
            max_value = 12,
            value = 1
        )

    monthdata2 = df[df["MONTH"]== minmonth].copy()
    monthdata2 = monthdata2.dropna(subset=["TMIN", "LATITUDE", "LONGITUDE"])
    tempdata2 = monthdata2.groupby(["STATION", "CITY", "LATITUDE", "LONGITUDE", "ELEVATION"])["TMIN"].mean().reset_index()

    mintemp2 = tempdata2["TMIN"].min()
    maxtemp2 = tempdata2["TMIN"].max()

    m3 = folium.Map(location=[38.6, -80.6], zoom_start=7, tiles=None, scrollWheelZoom=False)
    folium.TileLayer("OpenStreetMap", control=False).add_to(m3)

    for i, row in tempdata2.iterrows():

            tempvalue = (row["TMIN"] - mintemp2) / (maxtemp2 - mintemp2)

            tcolor = to_hex(matplotlib.colormaps["YlOrRd"](tempvalue))

            folium.CircleMarker(location = [row["LATITUDE"], row["LONGITUDE"]], 
                                radius = 5, color = tcolor, 
                                fill =True, 
                                fill_color = tcolor, 
                                fill_opacity = 1,
                                tooltip =f"City: {row['CITY']} | Avg. Min Temp.: {row['TMIN']:.1f} degrees F | Elevation: {row['ELEVATION']} M").add_to(m3)

    components.html(m3._repr_html_(), height=550)


    st.caption("Looking at the 3 maps, you can start to see a pattern.  The stations with higher elevation tend to record lower average highs and lower average lows per month.  This aligns with what meteorological studies have shown, that higher elevation tends to" \
    " relate to lower average temperatures.  It is also interesting to look specifically at the elevation map.  Picking out the darker red (higher elevation) stations you can start to vaguely see the path of mountain ranges in the state.  However, these maps only" \
    " provide a look into specific monthly average minimum and maximum temperatures.  In the next tab (Weather Over Time) we will examine how temperature and precipitation fluctuate over certain time periods")

#Time Series

with time:
    st.header("Weather Over Time")
    st.subheader("Temperature by Elevation Group")
    st.caption("Comparing monthly temperature patterns across 3 elevation groups in West Virginia")
    st.caption("In this plot we use the same temperature measures, average maximum and minimum monthly temperatures.  This time, we split the elevation into 3 equal width ranges in meters.")
    
    choice = st.selectbox("Select either Average Max or Min for Monthly Temperature", ["Max Temperature", "Min Temperature"])

    if choice == "Max Temperature":
         col = "TMAX"
    else:
         col = "TMIN"

    timedata = df.dropna(subset = [col, "MONTH", "El_class"]).copy()

    avg = timedata.groupby(["MONTH", "El_class"])[col].mean().reset_index()


    lowdata = avg[avg["El_class"]== "Low"]
    meddata = avg[avg["El_class"]== "Medium"]
    highdata = avg[avg["El_class"]== "High"]

    fig, ax = plt.subplots()

    ax.plot(lowdata["MONTH"], lowdata[col], label = "Low", color = "lightyellow")
    ax.plot(meddata["MONTH"], meddata[col], label = "Medium", color = "orange")
    ax.plot(highdata["MONTH"], highdata[col], label = "High", color = "darkred")

    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Temperature (F)")
    ax.set_title("Monthly Avg " + choice + " by Elevation")
    ax.legend(title = "Elevation Group")

    st.pyplot(fig)
    plt.close(fig)

    st.caption("Looking at the plot, we can confirm our suspicion that higher elevations are related to lower average monthly temperatures in 2025.  We can take this a step further, as past meteorological evidence suggests that this is the case.  Let's take a look at the precipitation over time for each elevation group.")

#Elevation and Rain
    st.subheader("Monthly Precipitation by Elevation Group")
    st.caption("Comparing monthly precipitation patterns across 3 elevation groups in West Virginia")
    st.caption("The precipitation in this plot is measured by daily average inches of rain for a given month.  Again, we use the same elevation level groupings.")
    
    rain = df.dropna(subset=["MONTH", "PRCP"]).copy()

    rain2 = rain.groupby(["MONTH", "El_class"])["PRCP"].mean().reset_index()

    lowdata2 = rain2[rain2["El_class"]=="Low"]
    meddata2 = rain2[rain2["El_class"]=="Medium"]
    highdata2 = rain2[rain2["El_class"]=="High"]

    months = st.slider("Rolling Window (months)", min_value = 1, max_value = 3, value = 1)

    fig, ax = plt.subplots()

    ax.plot(lowdata2["MONTH"], lowdata2["PRCP"].rolling(months).mean(), label = "Low Rolling Mean", color = "lightyellow")

    ax.plot(meddata2["MONTH"], meddata2["PRCP"].rolling(months).mean(), label = "Medium Rolling Mean", color = "orange")

    ax.plot(highdata2["MONTH"], highdata2["PRCP"].rolling(months).mean(), label = "High Rolling Mean", color = "darkred")

    ax.set_xlabel("Month")
    ax.set_ylabel("Avg. Daily Precipitation (in.)")
    ax.set_title("Monthly Avg. Precipitation by Elevation Group")
    ax.legend(title = "Elevation Group")

    st.pyplot(fig)
    plt.close(fig)

    st.caption("Looking at the precipitation by elevation plot, it appears that high elevation also relates to higher levels of precipitation.  This is especially noticeable when smoothing out the line by using a 3 month rolling mean in the slider.  " \
    "It is interesting that medium elevation tends to produce higher levels of precipitation, suggesting other geographic factors and differences may contribute to the pattern.  Given the sharp rise in precipitation during colder months for high and medium elevation, let's examine snowfall.")

#Winter Snowfall
    st.subheader("Winter Snow by Elevation Group")
    st.caption("Comparing monthly snow patterns across 3 elevation groups in West Virginia")
    st.caption("For the snowfall measurement, we use similar data to the precipitation plot.  The snowfall is measured in average daily snowfall in inches for a given month.")

    snow = df.dropna(subset=["MONTH", "SNOW", "El_class"]).copy()
    snow = snow[snow["MONTH"].isin([1,2,12])]
    snow2 = snow.groupby(["MONTH", "El_class"])["SNOW"].mean().reset_index()

    lowdata3 = snow2[snow2["El_class"]=="Low"]
    meddata3 = snow2[snow2["El_class"]=="Medium"]
    highdata3 = snow2[snow2["El_class"]=="High"]

    fig, ax = plt.subplots()

    ax.bar([1, 2, 3], lowdata3["SNOW"], width = .25, label = "Low", color = "lightyellow")
    ax.bar([1.25, 2.25, 3.25], meddata3["SNOW"], width = .25, label = "Medium", color = "orange")
    ax.bar([1.5, 2.5, 3.5], highdata3["SNOW"], width = .25, label = "High", color = "darkred")

    ax.set_xlabel("Month")
    ax.set_ylabel("Avg. Snowfall (in.)")
    ax.set_title("Monthly Avg. Snowfall by Elevation Group")
    ax.set_xticks([1.25, 2.25, 3.25], ["January", "February", "December"])
    ax.legend(title = "Elevation Group")

    st.pyplot(fig)
    plt.close(fig)

    st.caption("These bar charts suggest slightly different results for snowfall than for precipitation.  Here we can see the expected pattern according to meteorological records.  The highest elevation recorded the most snowfall, the medium elevation recorded the second most, and low elevation recorded the least." \
    "   Let's look at some other types of weather events using text analysis from reported events at each station next.")

with text:
    st.header("Weather Type Text Analysis")
    st.subheader("Frequency of Recorded Weather Types")
    st.caption("The data here comes from weather type reports, with each weather type being recorded in the dataset as Thunder, Smoke, etc. from each station it was observed from.  This data is for all of West Virginia's weather stations.")

    wtcols = ["WT01", "WT02", "WT03", "WT04", "WT05", "WT06", "WT08", "WT09", "WT11"]

    df["wtext"] = (df["WT01"].fillna("") + " " +
                          df["WT02"].fillna("") + " " +
                          df["WT03"].fillna("") + " " +
                          df["WT04"].fillna("") + " " +
                          df["WT05"].fillna("") + " " +
                          df["WT06"].fillna("") + " " +
                          df["WT08"].fillna("") + " " +
                          df["WT09"].fillna("") + " " +
                          df["WT11"].fillna("")

    )

    docs = df["wtext"]
    docs = docs[docs.str.strip() != ""]

    cv = CountVectorizer()
    counts = cv.fit_transform(docs)

    freq = pd.Series(counts.sum(axis=0).A1, index=cv.get_feature_names_out())

    fig, ax = plt.subplots(figsize = (10,5))

    freq.sort_values().plot.barh()
    ax.set_xlabel("Number of Observations")
    ax.set_ylabel("Weather Type")
    ax.set_title("Most Frequent Recorded Weather Types")
    ax.bar_label(ax.containers[0])



    st.pyplot(fig)
    plt.close(fig)

    st.caption("This bar chart shows that fog was by far the most reported weather event across West Virginia in 2025, followed by thunder, heavy fog, and smoke." \
    " We can take this a step further and examine how each weather event was reported at each elevation group.")

    #Low Elevation %
    st.subheader("Recorded Weather Types by Elevation")
    st.caption("This plot below shows the highest percentage of recorded weather types at low elevation.  The percentage was chosen as the amount of stations in each elevation level is not exactly equal.  This causes an imbalance of station records and reports, so percentage makes more sense for our exploration. Weather types come from each day of recorded weather for each station, and multiple weather types can occur on a single day.")

    lowdocs = df[df["El_class"]== "Low"]["wtext"].dropna()
    lowdocs = lowdocs[lowdocs.str.strip() != ""]

    lowcv = CountVectorizer()
    lowcounts= lowcv.fit_transform(lowdocs)

    lowfreq = pd.Series(lowcounts.sum(axis=0).A1, index=lowcv.get_feature_names_out())
    lowfreq = lowfreq/ len(df[df["El_class"] == "Low"]) *100
    lowfreq = lowfreq.round(1)

    fig, ax = plt.subplots()

    lowfreq.sort_values().plot.barh()
    ax.set_xlabel("Percentage of Observations")
    ax.set_ylabel("Weather Type")
    ax.set_title("Most Frequent Recorded Weather Types at Low Elev.")
    ax.bar_label(ax.containers[0])

    st.pyplot(fig)
    plt.close(fig)

#Med Elevation %
    st.caption("This plot below shows the highest percentage of recorded weather types at medium elevation.")

    meddocs = df[df["El_class"]== "Medium"]["wtext"].dropna()
    meddocs = meddocs[meddocs.str.strip() != ""]

    medcv = CountVectorizer()
    medcounts= medcv.fit_transform(meddocs)

    medfreq = pd.Series(medcounts.sum(axis=0).A1, index=medcv.get_feature_names_out())
    medfreq = medfreq/ len(df[df["El_class"] == "Medium"]) *100
    medfreq = medfreq.round(1)

    fig, ax = plt.subplots()

    medfreq.sort_values().plot.barh()
    ax.set_xlabel("Percentage of Observations")
    ax.set_ylabel("Weather Type")
    ax.set_title("Most Frequent Recorded Weather Types at Medium Elev.")
    ax.bar_label(ax.containers[0])

    st.pyplot(fig)
    plt.close(fig)

#High Elevation %
    st.caption("This plot below shows the highest percentage of recorded weather types at high elevation.")

    highdocs = df[df["El_class"]== "High"]["wtext"].dropna()
    highdocs = highdocs[highdocs.str.strip() != ""]

    highcv = CountVectorizer()
    highcounts= highcv.fit_transform(highdocs)

    highfreq = pd.Series(highcounts.sum(axis=0).A1, index=highcv.get_feature_names_out())
    highfreq = highfreq/ len(df[df["El_class"] == "High"]) *100
    highfreq = highfreq.round(1)

    fig, ax = plt.subplots()

    highfreq.sort_values().plot.barh()
    ax.set_xlabel("Percentage of Observations")
    ax.set_ylabel("Weather Type")
    ax.set_title("Most Frequent Recorded Weather Types at High Elev.")
    ax.bar_label(ax.containers[0])

    st.pyplot(fig)
    plt.close(fig)

    st.caption("The recorded weather event percentages for each elevation level appear to be relatively similar, with fog and thunder still being the most commonly recorded weather events in all 3 elevation groups." \
    "  One interesting takeaway is that high elevation recorded the highest percentage of high winds.  This makes sense when thinking about West Virginia's mountain ranges, and the strong windstorms that tend to occur there.")