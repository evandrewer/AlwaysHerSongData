import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import streamlit as st

@st.cache_data
def songdata(show_spinner=True, ttl=15):
    silhouette = pd.read_csv("Silhouette (The Halloween Song)-timeline.csv")
    itb = pd.read_csv("In the Beginning-timeline.csv")
    erberger = pd.read_csv("Airport Girl-timeline.csv")
    mr_nice_guy = pd.read_csv( "Mr. Nice Guy-timeline.csv")
    my_brain = pd.read_csv("My Brain is Carrying the World-timeline.csv")
    olay = pd.read_csv("One Look at You - Acoustic-timeline.csv")
    prolly_nun = pd.read_csv("Probably Nothing - Acoustic-timeline.csv")
    savior = pd.read_csv("Savior - Acoustic-timeline.csv")
    itb_acous = pd.read_csv("In the Beginning - Acoustic-timeline.csv")
    erberger_acous = pd.read_csv("Airport Girl - Acoustic-timeline.csv")
    timeless = pd.read_csv("Timeless-timeline.csv")

    silhouette['song'] = 'Silhouette'
    itb['song'] = 'In the Beginning'
    erberger['song'] = 'Airport Girl'
    mr_nice_guy['song'] = 'Mr. Nice Guy'
    my_brain['song'] = 'My Brain is Carrying the World'
    olay['song'] = 'One Look At You - Acoustic'
    prolly_nun['song'] = 'Probably Nothing - Acoustic'
    savior['song'] = 'Savior - Acoustic'
    itb_acous['song'] = 'In the Beginning - Acoustic'
    erberger_acous['song'] = 'Airport Girl - Acoustic'
    timeless['song'] = 'Timeless'

    combined_songs = pd.concat([silhouette, itb, erberger, mr_nice_guy, my_brain,
                            olay, prolly_nun, savior, itb_acous, erberger_acous, timeless])

    combined_songs['date'] = pd.to_datetime(combined_songs['date'])

    return combined_songs


# App code

song_data = songdata()


st.title("Always Her Spotify Stats")

song_titles = ['Silhouette', 'In the Beginning', 'Airport Girl', 'Mr. Nice Guy', 
               'My Brain is Carrying the World', 'One Look At You - Acoustic',
               'Probably Nothing - Acoustic', 'Savior - Acoustic',
               'In the Beginning - Acoustic', 'Airport Girl - Acoustic', 'Timeless']
selected_songs = st.sidebar.multiselect(
    "Select Songs", options=song_titles, default=song_titles)

data_by_song = song_data[song_data['song'].isin(selected_songs)]


# For tab1 col1
total_streams_per_song = (data_by_song.groupby('song', as_index=False)['streams']
                          .sum()) 
total_streams_per_song = total_streams_per_song.sort_values(by='streams', ascending=False)

grand_total = total_streams_per_song['streams'].sum() 


# For tab1 col2
data_by_song['cumulative_streams'] = data_by_song.groupby('song')['streams'].cumsum() 

def calculate_growth_rate(group, global_baseline):
    group = group.copy()
    group['avg_last_10_days'] = group['streams'].rolling(window=10, min_periods=1).mean()
    group['avg_prior_10_days'] = global_baseline.shift(10)

    # Calculate the growth rate between the two averages
    group['growth_rate'] = ((group['avg_last_10_days'] - group['avg_prior_10_days']) / group['avg_prior_10_days']) * 100
    return group

global_baseline = (
    data_by_song.groupby('date')['streams']
    .mean()
    .rolling(window=10, min_periods=1)
    .mean()
)

data_by_song = data_by_song.groupby('song', group_keys=False).apply(calculate_growth_rate, global_baseline)

growth_rate_per_song = (data_by_song
                        .dropna(subset=['growth_rate'])  # Remove NaN values
                        .groupby('song', as_index=False)
                        .agg({'growth_rate': 'last'})  # Take last non-null value
                        .sort_values(by='growth_rate', ascending=False))


# For tab1, col3
release_dates = (
    song_data[song_data["streams"] > 0] 
    .groupby("song")["date"]
    .min()
    .reset_index()
)
release_dates.columns = ["Song", "Release Date"]
release_dates["Release Date"] = pd.to_datetime(release_dates["Release Date"])

today = pd.to_datetime("today")
release_dates["days_since_release"] = (today - release_dates["Release Date"]).dt.days

filtered_release_df = release_dates[release_dates["Song"].isin(selected_songs)]
filtered_release_df = filtered_release_df.drop(columns=["Release Date"]).set_index("Song")
filtered_release_df = filtered_release_df.sort_values(by='days_since_release', ascending=False)


# For tab1, col4
scatter_data = total_streams_per_song.merge(
    release_dates[["Song", "days_since_release"]],
    left_on="song",
    right_on="Song",
    how="left"
).drop(columns=["Song"]) 

plt.style.use('dark_background')
colors = plt.cm.get_cmap("tab20", len(scatter_data))



tab1, tab2 = st.tabs(['General Stats', 'Cumulative Weekly Streams'])

with tab1:

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Total Streams")
        total_streams_per_song = total_streams_per_song.rename(columns={'song': 'Song', 'streams': 'Streams'})
        st.dataframe(total_streams_per_song, hide_index=True, use_container_width=True, height = 422)
        st.write(f"**Grand Total Streams**: {grand_total}")

    with col2:
        st.subheader("Days Since Release")
        filtered_release_df = filtered_release_df.rename(columns={'song': 'Song', 'days_since_release': 'Days'})
        st.dataframe(filtered_release_df, hide_index=False, use_container_width=True, height = 422)

    col3 = st.columns([1])[0]

    with col3:
        st.subheader("Days Since Release vs. Total Streams")
        fig, ax = plt.subplots()

        for idx, (song, days, streams) in enumerate(zip(scatter_data["song"], scatter_data["days_since_release"], scatter_data["streams"])):
            ax.scatter(days, streams, color=colors(idx), label=song, s=15, alpha=0.8)  

        # Trendline Calculation
        x = scatter_data["days_since_release"]
        y = scatter_data["streams"]
        
        if len(x) > 1:
            coeffs = np.polyfit(x, y, deg=1)
            trendline = np.poly1d(coeffs)  
            sorted_x = np.sort(x)
            ax.plot(sorted_x, trendline(sorted_x), linestyle="dashed", color="white", alpha=0.7, linewidth=0.8, label="Trendline")

        # Labels & Formatting
        ax.set_xlabel("Days Since Release", color='white')
        ax.set_ylabel("Total Streams", color='white')
        ax.grid(True, linestyle='--', color='gray', alpha=0.5)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)

        st.pyplot(fig)

    col4 = st.columns([1])[0]

    with col4:
        st.subheader("10-day Growth Rate")
        growth_rate_per_song = growth_rate_per_song.rename(columns={'song': 'Song', 'growth_rate': 'Growth Rate %'})
        st.dataframe(growth_rate_per_song, hide_index=True, use_container_width=True, height = 423)