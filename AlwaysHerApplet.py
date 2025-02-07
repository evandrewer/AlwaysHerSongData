import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import streamlit as st

@st.cache_data
def song_data():
    root = "C:/Users/HUBFU/OneDrive/Documents/Always Her song data/"

    # Read in song data csv files
    silhouette = pd.read_csv(root + "Silhouette (The Halloween Song)-timeline.csv")
    itb = pd.read_csv(root + "In the Beginning-timeline.csv")
    erberger = pd.read_csv(root + "Airport Girl-timeline.csv")
    mr_nice_guy = pd.read_csv(root + "Mr. Nice Guy-timeline.csv")
    my_brain = pd.read_csv(root + "My Brain Is Carrying the World-timeline.csv")
    olay = pd.read_csv(root + "One Look at You - Acoustic-timeline.csv")
    prolly_nun = pd.read_csv(root + "Probably Nothing - Acoustic-timeline.csv")
    savior = pd.read_csv(root + "Savior - Acoustic-timeline.csv")
    itb_acous = pd.read_csv(root + "In the Beginning - Acoustic-timeline.csv")
    erberger_acous = pd.read_csv(root + "Airport Girl - Acoustic-timeline.csv")
    timeless = pd.read_csv(root + "Timeless-timeline.csv")

    silhouette['song'] = 'Silhouette'
    itb['song'] = 'In the Beginning'
    erberger['song'] = 'Airport Girl'
    mr_nice_guy['song'] = 'Mr. Nice Guy'
    my_brain['song'] = 'My Brain Is Carrying the World'
    olay['song'] = 'One Look At You'
    prolly_nun['song'] = 'Probably Nothing (Acoustic)'
    savior['song'] = 'Savior (Acoustic)'
    itb_acous['song'] = 'In the Beginning (Acoustic)'
    erberger_acous['song'] = 'Airport Girl (Acoustic)'
    timeless['song'] = 'Timeless'

    combined_songs = pd.concat([silhouette, itb, erberger, mr_nice_guy, my_brain,
                            olay, prolly_nun, savior, itb_acous, erberger_acous, timeless])

    combined_songs['date'] = pd.to_datetime(combined_songs['date'])

    return combined_songs


st.title("Always Her Spotify Stats")