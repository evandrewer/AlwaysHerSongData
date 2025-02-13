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
song_summary = (data_by_song[data_by_song["streams"] > 0]
                .groupby('song', as_index=False)
                .agg(Streams=('streams', 'sum'),
                     Release_Date=('date', 'min')))

song_summary["Release_Date"] = pd.to_datetime(song_summary["Release_Date"])
today = pd.to_datetime("today")
song_summary["Days"] = (today - song_summary["Release_Date"]).dt.days

song_summary["streams_per_day"] = song_summary["Streams"] / song_summary["Days"]

song_summary = song_summary.sort_values(by='Streams', ascending=False)

grand_total = song_summary["Streams"].sum()


# For tab1, col2
scatter_data = song_summary.copy()

plt.style.use('dark_background')
colors = plt.cm.get_cmap("tab20", len(scatter_data))


# For tab1 col3

def calculate_growth_rate(group):
    group['avg_last_10_days'] = group['streams'].rolling(window=10, min_periods=1).mean()
    group['avg_prior_10_days'] = group['streams'].shift(10).rolling(window=10, min_periods=1).mean()
    
    group['growth_rate'] = ((group['avg_last_10_days'] - group['avg_prior_10_days']) / group['avg_prior_10_days']) * 100

    return group

data_by_song = data_by_song.groupby('song', group_keys=False).apply(calculate_growth_rate)

growth_rate_per_song = (data_by_song.dropna(subset=['growth_rate'])  # Remove NaN values
                        .groupby('song', as_index=False)
                        .agg({'growth_rate': 'last'})  # Take last non-null value
                        .sort_values(by='growth_rate', ascending=False))




tab1, tab2 = st.tabs(['General Stats', 'Stream Breakdowns'])

with tab1:

    col1 = st.columns([1])[0]

    with col1:
        st.subheader("Total Streams & Days Since Release")
        song_summary = song_summary.rename(columns={'song': 'Song', 'Days': 'Days Since Release', 'streams_per_day': 'Streams Per Day' })
        song_summary_col1 = song_summary.drop(columns='Release_Date')
        st.data_editor(song_summary_col1, hide_index=True, use_container_width=True, height=422)
        st.write(f"**Grand Total Streams**: {grand_total}")

    col2 = st.columns([1])[0]

    with col2:
        st.subheader("Days Since Release vs. Streams")
        fig, ax = plt.subplots()

        for idx, (song, days, streams) in enumerate(zip(scatter_data["song"], scatter_data["Days"], scatter_data["Streams"])):
            ax.scatter(days, streams, color=colors(idx), label=song, s=15, alpha=0.8)  

        # Trendline Calculation
        x = scatter_data["Days"]
        y = scatter_data["Streams"]
        
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

    col3 = st.columns([1])[0]

    with col3:
        st.subheader("10-day Moving Growth Rate")
        growth_rate_per_song = growth_rate_per_song.rename(columns={'song': 'Song', 'growth_rate': 'Growth Rate %'})
        st.dataframe(growth_rate_per_song, hide_index=True, use_container_width=True, height = 423)

        with st.expander("See explanation"):
            st.write("""
                The 10-day moving growth rate is calculated by taking the average streams of a given song for 
                the past 10 days and comparing to its own average streams for the 10 days prior
                to that. For example, if a song has a 10-day growth rate of 15%, that means it was streamed
                about 15% more than in the 10 day period before.
                    """)



    with tab2:

        st.subheader("Average Daily Streams per Song")
        view_option = st.radio("Select View", ["Daily Average Streams", "Weekly Average Streams"])

        earliest_release_date = song_summary["Release_Date"].min()
        filtered_data = data_by_song[data_by_song["date"] >= earliest_release_date].copy()

        daily_avg_streams = (filtered_data
                            .groupby("date")["streams"]
                            .mean()
                            .reset_index()
                            .rename(columns={"streams": "Daily Streams"}))

        weekly_avg_streams = (daily_avg_streams
                            .groupby(pd.Grouper(key="date", freq="W"))["Daily Streams"]
                            .sum()
                            .reset_index()
                            .rename(columns={"Daily Streams": "Weekly Streams"}))

        # Decide which data to plot based on the selected view option
        if view_option == "Weekly Average Streams":
            plot_data = weekly_avg_streams
            y_column = "Weekly Streams"
        else:
            plot_data = daily_avg_streams
            y_column = "Daily Streams"

        # Create the line chart for the chosen streams
        st.line_chart(plot_data.set_index("date")[[y_column]], use_container_width=True, color="#1DB954")


    # --- PIE CHART FOR STREAM DISTRIBUTION ---

        st.subheader("Song Stream Distribution on a Specific Day")

        # Date selector
        selected_date = st.date_input("Select a date", min_value=earliest_release_date, max_value=data_by_song["date"].max())

        # Filter data for selected date
        selected_day_data = data_by_song[data_by_song["date"] == pd.to_datetime(selected_date)]

        if not selected_day_data.empty:
            # Compute total streams per song for that date
            stream_distribution = (selected_day_data
                                    .groupby("song")["streams"]
                                    .sum()
                                    .reset_index())

            # Remove songs with 0 streams
            stream_distribution = stream_distribution[stream_distribution["streams"] > 0]

            # Calculate the total streams for the selected day
            total_streams = stream_distribution["streams"].sum()

            if not stream_distribution.empty:
                # Custom label function to show both percentage and stream count
                def autopct_format(pct, all_vals):
                    total = sum(all_vals)
                    absolute = int(round(pct * total / 100.0))  # Convert percentage to stream count
                    return f"{pct:.1f}%\n({absolute})"

                # Create a pie chart
                fig, ax = plt.subplots()
                wedges, texts, autotexts = ax.pie(
                    stream_distribution["streams"], 
                    labels=stream_distribution["song"], 
                    autopct=lambda pct: autopct_format(pct, stream_distribution["streams"]),
                    startangle=90, 
                    colors=plt.cm.tab20.colors
                )

                # Style the text
                for text in texts:
                    text.set_fontsize(10)
                    text.set_fontweight("bold")
                for autotext in autotexts:
                    autotext.set_fontsize(9)

                ax.axis("equal")  # Equal aspect ratio ensures the pie chart is circular

                st.pyplot(fig)

                # Display total stream count below the pie chart
                st.markdown(f"**Total Streams:** {total_streams:,}")
            else:
                st.write("No songs had streams on this date.")
        else:
            st.write("No stream data available for this date.")
