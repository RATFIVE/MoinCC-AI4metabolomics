import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import subprocess
import os
import sys
import psutil  # For checking running processes
from pathlib import Path
from PIL import Image
import lorem
from classes import *
from example_pic import plot_picture


loaddata = LoadData()

peakfinder = PeakFinder()

st.set_page_config(layout="wide", page_title="MoinCC - Application", page_icon=":shark:")

class StreamlitApp():

    def __init__(self, fig1=None, fig2=None):
        self.fig1 = fig1
        self.fig2 = fig2

    def side_bar(self):
        st.sidebar.title('How to')
        st.sidebar.markdown(
            """
            1. Upload the .CSV-File 
            2. ...
            3. ...

            """)
    
                        
    def header(self):
        st.markdown(
            """
                # MoinCC - Application
            """
        )
        st.file_uploader("Upload a the Metabolite Spectrum CSV")

        main, about = st.tabs(['Main Page', 'About'])

        with main:
            st.markdown('# Main')

        with about:
            st.markdown('# About')

    
        
        



    def run(self):
        self.side_bar()
        self.header()
        


# ----------------------------------------------------------------------------------------------------




if __name__ == '__main__':


    file = str(Path('FA_20231123_2H Yeast_Fumarate-d2_12 .csv'))
    substrates = loaddata.get_substrate_list(file)
    metabolites = loaddata.get_metabolite_list(file)
    df_list = loaddata.load_data(file)
    df = pd.read_csv(df_list[0])
    fig2 = plot_picture(df=df)

    example_image_path = Path(r'/Users/marco/Documents/MoinCC-AI4metabolomics/05_Streamlit/example/FA_20240228_2H_yeast_fumarate-d2_4.csv_output/FA_20240228_2H_yeast_fumarate-d2_4.csv_time_dependence.png')
    
    # Run Streamlit App
    app = StreamlitApp(fig1=example_image_path, fig2=fig2)
    app.run()


