import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import matplotlib.colors as mcolors

import streamlit as st

@st.cache_data
def songdata(show_spinner=True):
    st.cache_data.clear()
    
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
    backnforth = pd.read_csv("Back & Forth-timeline.csv")
    mb8bit = pd.read_csv("My Brain is Carrying the World - 8-Bit Remix-timeline.csv")

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
    backnforth['song'] = 'Back & Forth'
    mb8bit['song'] = 'My Brain (8-bit)'

    combined_songs = pd.concat([silhouette, itb, erberger, mr_nice_guy, my_brain,
                            olay, prolly_nun, savior, itb_acous, erberger_acous, timeless, 
                            backnforth, mb8bit])

    combined_songs['date'] = pd.to_datetime(combined_songs['date'])

    return combined_songs


# App font change

st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
        }
        
        /* Change font for headers */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Verdana', sans-serif;
        }

        /* Change font for sidebar */
        .sidebar .sidebar-content {
            font-family: 'Courier New', monospace;
        }

        /* Change font for plot titles and legends */
        .plot-container {
            font-family: 'Trebuchet MS', sans-serif;
        }
    </style>
    """,
    unsafe_allow_html=True
)



# App code

song_data = songdata()
song_data2 = song_data

st.title("Always Her Streaming Stats")

st.sidebar.header("Select Data Sources")
source_mapping = {
    "Spotify": "spotify",
    "Apple Music": "apple",
    "YouTube Music": "youtube",
    "Amazon Music": "amazon",
    "Other": "other"
}

selected_sources = st.sidebar.multiselect(
    "Choose Streaming Data:",
    options=list(source_mapping.keys()),
    default=["Spotify", "Apple Music", "YouTube Music", "Amazon Music", "Other"]  # Default selection
)

# Ensure at least one source is selected
if not selected_sources:
    st.warning("Please select at least one streaming source.")
else:
    # Sum the selected sources using the correct variable names
    song_data["selected_streams"] = sum(song_data[source_mapping[source]] for source in selected_sources)



# Song selection multiselect
song_titles = ['Silhouette', 'In the Beginning', 'Airport Girl', 'Mr. Nice Guy', 
               'My Brain is Carrying the World', 'One Look At You - Acoustic',
               'Probably Nothing - Acoustic', 'Savior - Acoustic',
               'In the Beginning - Acoustic', 'Airport Girl - Acoustic', 'Timeless',
               'Back & Forth', 'My Brain (8-bit)']

selected_songs = st.sidebar.multiselect(
    "Select Songs", options=song_titles, default=song_titles)

# Filter dataset based on selected songs
data_by_song = song_data[song_data['song'].isin(selected_songs)]


# Color dictionary

tab20 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a'] #add tab20 red (light) next

tab20_colors = mcolors.ListedColormap(tab20)

color_dict = {song: tab20_colors(i) for i, song in enumerate(song_titles)}




# For tab1, col1
song_summary = (data_by_song[data_by_song['selected_streams'] > 0]
                .groupby('song', as_index=False)
                .agg(Streams=('selected_streams', 'sum'),
                     Release_Date=('date', 'min')))

song_summary["Release_Date"] = pd.to_datetime(song_summary["Release_Date"])
today = pd.to_datetime("today")
song_summary["Days"] = (today - song_summary["Release_Date"]).dt.days - 1 # Subtracting one to account for the current day, which streaming data has not arrived for yet

song_summary["streams_per_day"] = song_summary["Streams"] / song_summary["Days"]

song_summary = song_summary.sort_values(by='Streams', ascending=False)

grand_total = song_summary["Streams"].sum()


# For tab1, col2

min_valid_date = song_data[song_data[["spotify", "apple", "youtube", "amazon", "other"]].sum(axis=1) > 0]["date"].min()


# For tab1, col3
scatter_data = song_summary.copy()

plt.style.use('dark_background')
colors = tab20_colors


# For tab1, col4

def calculate_growth_rate_and_proportion(group):
    group['avg_last_10_days'] = group['selected_streams'].rolling(window=10, min_periods=1).mean()
    group['avg_prior_10_days'] = group['selected_streams'].shift(10).rolling(window=10, min_periods=1).mean()

    group['growth_rate'] = ((group['avg_last_10_days'] - group['avg_prior_10_days']) / group['avg_prior_10_days']) * 100

    return group

data_by_song = data_by_song.groupby('song', group_keys=False).apply(calculate_growth_rate_and_proportion)
data_by_song['daily_stream_proportion'] = data_by_song['selected_streams'] / data_by_song.groupby('date')['selected_streams'].transform('sum')

data_by_song = data_by_song.reset_index(drop=True)  # Ensure unique index values
data_by_song['avg_10_day_proportion'] = (
    data_by_song.groupby('song')['daily_stream_proportion']
    .rolling(window=10, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

data_by_song['avg_10_day_proportion'] *= 100

growth_rate_per_song = (data_by_song.dropna(subset=['growth_rate'])  
                        .groupby('song', as_index=False)
                        .agg({'growth_rate': 'last', 'avg_10_day_proportion': 'last'})  
                        .sort_values(by='growth_rate', ascending=False))




tab1, tab2, tab3 = st.tabs(['General Stats', 'Stream Breakdowns', 'Song Comparisons'])

with tab1:

    col1 = st.columns([1])[0]

    with col1:
        st.subheader("Total Streams & Days Since Release")
        song_summary = song_summary.rename(columns={'song': 'Song', 'Days': 'Days Since Release', 'streams_per_day': 'Streams Per Day' })
        song_summary_col1 = song_summary.drop(columns='Release_Date')
        st.data_editor(song_summary_col1, hide_index=True, use_container_width=True, height=491)
        st.write(f"**Grand Total Streams**: {grand_total}")

    col2 = st.columns([1])[0]

    with col2:

        start_date_t1, end_date_t1 = st.date_input(
            "Select Date Range:",
            value=[min_valid_date, song_data["date"].max()],  # Default to full range
            min_value=min_valid_date,
            max_value=song_data["date"].max()
        )

        filtered_data_pie = song_data[(song_data["date"] >= pd.to_datetime(start_date_t1)) & 
                                (song_data["date"] <= pd.to_datetime(end_date_t1))]

        total_spotify = filtered_data_pie["spotify"].sum()
        total_apple = filtered_data_pie["apple"].sum()
        total_youtube = filtered_data_pie["youtube"].sum()
        total_amazon = filtered_data_pie["amazon"].sum()
        total_other = filtered_data_pie["other"].sum()

        platforms = ["Spotify", "Apple Music", "YouTube", "Amazon Music", "Other"]
        stream_counts = [total_spotify, total_apple, total_youtube, total_amazon, total_other]


        fig = px.pie(
            names=platforms,
            values=stream_counts,
            title=f"Proportion of Streams by Platform ({start_date_t1} to {end_date_t1})",
            color=platforms,
            color_discrete_map={
                "Spotify": "#1DB954", 
                "Apple Music": "#F52F45", 
                "YouTube": "#FF0000", 
                "Amazon Music": "#25D1DA",
                "Other": "808080"
            }
        )

        # Disable hover text
        fig.update_traces(hoverinfo="skip")

        # Display chart
        st.plotly_chart(fig)


    col3 = st.columns([1])[0]

    with col3:
        st.subheader("Days Since Release vs. Streams")
        fig, ax = plt.subplots()

        for idx, (song, days, streams) in enumerate(zip(scatter_data["song"], scatter_data["Days"], scatter_data["Streams"])):
            ax.scatter(days, streams, color=color_dict[song], label=song, s=15, alpha=0.8)  

        ax.scatter(0, 0, color="white", alpha=0)  

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

    col4 = st.columns([1])[0]

    with col4:
        st.subheader("10-day Moving Growth Rate and Avg Stream Proportion")
        growth_rate_per_song = growth_rate_per_song.rename(columns={'song': 'Song', 'growth_rate': 'Growth Rate %', 'avg_10_day_proportion': 'Average Proportion %'})
        st.dataframe(growth_rate_per_song, hide_index=True, use_container_width=True, height = 491)

        with st.expander("See explanation"):
            st.write("""
                The 10-day moving growth rate is calculated by taking the average streams of a given song for 
                the past 10 days and comparing to its own average streams for the 10 days prior
                to that. For example, if a song has a 10-day growth rate of 15%, that means it was streamed
                about 15% more than in the 10 day period before.
                     
                The 10-day average stream proportion is calculated by taking the proportion of streams for each 
                song each day for 10 days, and averaging across days.
                     
                The "Other" selection includes the following services: Pandora, Deezer, and Boomplay.
                    """)



    with tab2:

        st.subheader("Total Streams")
        view_option = st.radio("Select View", ["Daily Streams", "Weekly Streams"])

        earliest_release_date = song_summary["Release_Date"].min()
        filtered_data = data_by_song[data_by_song["date"] >= earliest_release_date].copy()

        daily_avg_streams = (filtered_data
                            .groupby("date")['selected_streams']
                            .sum()
                            .reset_index()
                            .rename(columns={'selected_streams': "Daily Streams"}))

        weekly_avg_streams = (daily_avg_streams
                            .groupby(pd.Grouper(key="date", freq="W"))["Daily Streams"]
                            .sum()
                            .reset_index()
                            .rename(columns={"Daily Streams": "Weekly Streams"}))

        # Decide which data to plot based on the selected view option
        if view_option == "Weekly Streams":
            plot_data = weekly_avg_streams
            y_column = "Weekly Streams"
        else:
            plot_data = daily_avg_streams
            y_column = "Daily Streams"

        # Create the line chart for the chosen streams
        st.line_chart(plot_data.set_index("date")[[y_column]], use_container_width=True, color="#1DB954")


    # --- PIE CHART FOR STREAM DISTRIBUTION ---

        # Date selector
        selected_date = st.date_input("Select a date", min_value=earliest_release_date, max_value=data_by_song["date"].max())

        # Filter data for selected date
        selected_day_data = data_by_song[data_by_song["date"] == pd.to_datetime(selected_date)]

        if not selected_day_data.empty:
            # Compute total streams per song for that date
            stream_distribution = (selected_day_data
                                    .groupby("song")['selected_streams']
                                    .sum()
                                    .reset_index())

            # Remove songs with 0 streams
            stream_distribution = stream_distribution[stream_distribution['selected_streams'] > 0]

            if not stream_distribution.empty:
                # Convert Matplotlib RGBA colors to HEX for Plotly
                def rgba_to_hex(rgba):
                    return mcolors.to_hex(rgba)

                plotly_color_dict = {song: rgba_to_hex(color) for song, color in color_dict.items()}

                # Calculate total streams for the selected date
                total_streams = stream_distribution['selected_streams'].sum()

                # Add custom labels showing both percentage and total streams
                stream_distribution["label_text"] = stream_distribution.apply(
                    lambda row: f"{(row['selected_streams'] / total_streams) * 100:.1f}%\n({row['selected_streams']:,})", 
                    axis=1
                )

                # Create a Plotly pie chart
                fig = px.pie(
                    stream_distribution,
                    names="song",
                    values="selected_streams",
                    title=f"Stream Distribution on {selected_date.strftime('%B %d, %Y')}",
                    color="song",
                    color_discrete_map=plotly_color_dict,  # Use converted colors
                )

                # Display labels directly on the pie chart
                fig.update_traces(
                    text=stream_distribution["label_text"],  # Show both percentage & total streams
                    textinfo="text",  # Display custom labels directly
                    hoverinfo="skip"  # Disable hover text
                )

                st.plotly_chart(fig)

                # Display total stream count below the pie chart
                st.markdown(f"**Total Streams on {selected_date.strftime('%B %d, %Y')}:** {total_streams:,}")

            else:
                st.write("No songs had streams on this date.")
        else:
            st.write("No stream data available for this date.")


    # STACKED BAR CHART

        st.subheader("Weekly Stream Counts Stacked Bar Graph")

        data_by_song = data_by_song[data_by_song['date'] >= '2024-04-21']

        data_by_song['week'] = data_by_song['date'].dt.to_period('W').apply(lambda r: r.start_time)
        weekly_df = data_by_song.groupby(['song', 'week'])['selected_streams'].sum().reset_index()
        pivot_df = weekly_df.pivot(index='week', columns='song', values='selected_streams').fillna(0)
        pivot_df = pivot_df.sort_index()


        # Reorder pivot_df columns based on first stream date
        pivot_df = pivot_df.reindex(columns=selected_songs)


        fig, ax = plt.subplots(figsize=(12, 6))

        bottom = pd.Series([0] * len(pivot_df), index=pivot_df.index)

        # Create the stacked bar chart
        for idx, song in enumerate(pivot_df.columns):
            ax.bar(pivot_df.index, pivot_df[song], 
                bottom=bottom, 
                label=song,
                color=color_dict[song], 
                width=4.5)
            bottom += pivot_df[song]

        # Formatting
        ax.set_title('Weekly Stream Counts (Stacked)', fontsize=16)
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Total Streams', fontsize=12)
        ax.legend(title='Song', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, which='both', alpha=0.3)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)


    # CUMU WEEKLY STREAMS CHART

        st.subheader("Cumulative Weekly Stream Counts")

        # Adjust df
        def filter_zeros_before_release(df):
            filtered_dfs = []
            for song, song_data in df.groupby('song'):
                song_data = song_data.sort_values('week').reset_index(drop=True)
                first_stream_week_idx = song_data.loc[song_data['selected_streams'] > 0].index.min()
                prior_week_idx = max(first_stream_week_idx - 1, 0)
                filtered_song_data = song_data.loc[prior_week_idx:]
                filtered_dfs.append(filtered_song_data)
            return pd.concat(filtered_dfs)
        filtered_df2 = filter_zeros_before_release(weekly_df)

        cumulative_df = filtered_df2.groupby('song').apply(
            lambda x: x.sort_values('week').assign(cumulative_streams=x['selected_streams'].cumsum())
        ).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, (song, song_data) in enumerate(cumulative_df.groupby('song')):
            ax.plot(song_data['week'], song_data['cumulative_streams'], 
                    label=song, 
                    color=color_dict.get(song, "gray"),  # Use color_dict, default to gray if song not found
                    linewidth=1)                                    

        ax.set_title('Cumulative Weekly Stream Counts', fontsize=16)
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Cumulative Streams', fontsize=12)
        ax.legend(title='Song', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', color='gray', alpha=0.7)  # Darker, dashed grid lines
        plt.tight_layout()

        st.pyplot(fig)


    # Last 28 Days Plot

        st.subheader("Stream Counts for Past 28 Days")

        latest_date = data_by_song['date'].max()
        past_28_days_df = data_by_song[data_by_song['date'] >= (latest_date - timedelta(days=28))]

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, (song, song_data) in enumerate(past_28_days_df.groupby('song')):
            ax.plot(song_data['date'], song_data['selected_streams'], 
                    label=song, 
                    color=color_dict.get(song, "gray"),  # Ensuring consistency with the defined color scheme
                    linewidth=1)                                    

        ax.set_title('Stream Counts for Past 28 Days', fontsize=16)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Streams', fontsize=12)
        #ax.set_xticklabels(song_data['date'], rotation=45)  # Ensuring readability of x-axis labels
        ax.legend(title='Song', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', color='gray', alpha=0.7)  # Standardizing grid appearance for visual clarity
        plt.tight_layout()

        st.pyplot(fig)



    with tab3:

        col1 = st.columns([1])[0]

        with col1:

            start_date, end_date = st.date_input(
                "Select Date Range:",
                value=[song_data["date"].max() - timedelta(days=28), song_data["date"].max()],
                min_value=song_data2["date"].min(),
                max_value=song_data2["date"].max(),
                key='date1'
            )

            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)


        col2, col3 = st.columns([1,1])


        with col2:

            selected_song = st.selectbox("Select a Song:", song_data2["song"].unique(), key='song1')

            filtered_data2 = song_data2[
                (song_data2["song"] == selected_song) & 
                (song_data2["date"] >= start_date) & 
                (song_data2["date"] <= end_date)
            ]

            # Sum streams from each platform within the date range
            stream_sums = {
                "Spotify": filtered_data2["spotify"].sum(),
                "Apple Music": filtered_data2["apple"].sum(),
                "YouTube": filtered_data2["youtube"].sum(),
                "Amazon Music": filtered_data2["amazon"].sum(),
                "Other": filtered_data2["other"].sum()
            }

            # Remove platforms with zero streams to avoid empty slices
            stream_sums = {k: v for k, v in stream_sums.items() if v > 0}

            stream_df = pd.DataFrame({
                "Platform": stream_sums.keys(),
                "Streams": stream_sums.values()
            })

            total_streams = sum(stream_sums.values())

            # Add custom labels showing both percentage and total streams
            stream_df["label_text"] = stream_df.apply(
                lambda row: f"{(row['Streams'] / total_streams) * 100:.1f}%\n({row['Streams']:,})", 
                axis=1
            )


            # ADD OTHER STATS HERE


            # Create the Plotly pie chart
            fig = px.pie(
                stream_df,
                names="Platform",
                values="Streams",
                title=f"Stream Sources for {selected_song}",
                color="Platform",
                color_discrete_map={
                    "Spotify": "#1DB954",
                    "Apple Music": "#F52F45",
                    "YouTube": "#FF0000",
                    "Amazon Music": "#25D1DA",
                    "Other": "808080"
                },
            )

            # subtitle
            fig.add_annotation(
                text=f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
                y=1.03,
                x=0.9,
                showarrow=False
            )

            # shrink legend and add margins
            fig.update_layout(
                margin=dict(t=30, b=0, l=80, r=0),
                height=350,
                width=350
            )

            # Display labels directly on the pie chart
            fig.update_traces(
                text=stream_df["label_text"],  # Show both percentage & total streams
                textinfo="text",  # Display custom labels directly
                hoverinfo="skip"  # Disable hover text
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, key='plot1')


        with col3:

            selected_song2 = st.selectbox("Select a Song:", song_data2["song"].unique(), key='song2', index = 1)


            filtered_data2 = song_data2[
                (song_data2["song"] == selected_song2) & 
                (song_data2["date"] >= start_date) & 
                (song_data2["date"] <= end_date)
            ]

            # Sum streams from each platform within the date range
            stream_sums = {
                "Spotify": filtered_data2["spotify"].sum(),
                "Apple Music": filtered_data2["apple"].sum(),
                "YouTube": filtered_data2["youtube"].sum(),
                "Amazon Music": filtered_data2["amazon"].sum(),
                "Other": filtered_data2["other"].sum()
            }

            # Remove platforms with zero streams to avoid empty slices
            stream_sums = {k: v for k, v in stream_sums.items() if v > 0}

            stream_df = pd.DataFrame({
                "Platform": stream_sums.keys(),
                "Streams": stream_sums.values()
            })

            total_streams = sum(stream_sums.values())

            # Add custom labels showing both percentage and total streams
            stream_df["label_text"] = stream_df.apply(
                lambda row: f"{(row['Streams'] / total_streams) * 100:.1f}%\n({row['Streams']:,})", 
                axis=1
            )


             # ADD OTHER STATS HERE


            # Create the Plotly pie chart
            fig2 = px.pie(
                stream_df,
                names="Platform",
                values="Streams",
                title=f"Stream Sources for {selected_song2}",
                color="Platform",
                color_discrete_map={
                    "Spotify": "#1DB954",
                    "Apple Music": "#F52F45",
                    "YouTube": "#FF0000",
                    "Amazon Music": "#25D1DA",
                    "Other": "808080"
                }
            )

            # subtitle
            fig2.add_annotation(
                text=f"({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
                y=1.03,
                x=0.9,
                showarrow=False
            )

            # shrink legend and add margins
            fig2.update_layout(
                margin=dict(t=30, b=0, l=80, r=0),
                height=350,
                width=350
            )

            # Display labels directly on the pie chart
            fig2.update_traces(
                text=stream_df["label_text"],  # Show both percentage & total streams
                textinfo="text",  # Display custom labels directly
                hoverinfo="skip"  # Disable hover text
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig2, key='plot2')