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

    def __init__(self, 
                 fig1=None, 
                 fig2=None,
                 fig3=None,
                 fig4=None,
                 fig5=None
                 ):
        
        self.fig1 = fig1
        self.fig2 = fig2
        self.fig3 = fig3
        self.fig4 = fig4
        self.fig5 = fig5



    def side_bar(self):
        st.sidebar.title('How to')
        
        st.sidebar.markdown(
            """
            1. Upload the .CSV-File 
            2. Use the buttons to navigate panels
            3. Explore the analysis
            """
        )
        

    def header(self):
        st.markdown("""<h1 style="text-align: center;">MoinCC - Application</h1>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
        with col2:
            st.file_uploader("Upload the Metabolite Spectrum CSV")
        main, about = st.tabs(['Main Page', 'About'])
        self.main_page(main)
        self.about_page(about)
    
    def main_page(self, main):
        with main:
            self.panel1()
            self.panel2()
            self.panel3()
            self.panel4()
            self.panel5()
        return None

    def about_page(self, about):
        with about:
            st.markdown(f"""
                        # About

                        This is a descrption of {lorem.paragraph(), lorem.paragraph()}
                        
                        """)
        return None

    # -------- Panels with Expanders --------------------
    def panel1(self):
        
        with st.expander("Panel 1 - Substrate Plot", expanded=True):
            st.markdown('# Substrate Plot')
            st.write("Content for Panel 1.")
        return None

    def panel2(self):
        
        with st.expander("Panel 2 - Kinetic Plot", expanded=True):
            st.markdown('# Kinetic Plot')
            st.image(self.fig1, caption="Your Image Caption", use_column_width=True)

        return None

    def panel3(self):
        
        with st.expander("Panel 3 - Kinetic Plot", expanded=True):
            st.markdown('# Panel 3')
            
        return None
    
    def panel4(self):
        
        with st.expander("Panel 4", expanded=True):
            st.markdown('# Panel 4')
        return None
    
    def panel5(self):
        
        with st.expander("Panel 5", expanded=True):
            st.markdown('# Panel 5')

        return None
    
    def run(self):
        #self.side_bar()
        self.header()

        return None
    
        
        


# ----------------------------------------------------------------------------------------------------




if __name__ == '__main__':


    file = str(Path('FA_20231123_2H Yeast_Fumarate-d2_12 .csv'))
    substrates = loaddata.get_substrate_list(file)
    metabolites = loaddata.get_metabolite_list(file)
    df_list = loaddata.load_data(file)
    df = pd.read_csv(df_list[0])
    fig2 = plot_picture(df=df)

    example_image_path = str(Path(r'/Users/marco/Documents/MoinCC-AI4metabolomics/05_Streamlit/example/FA_20240228_2H_yeast_fumarate-d2_4.csv_output/FA_20240228_2H_yeast_fumarate-d2_4.csv_time_dependence.png'))
    
    # Run Streamlit App
    app = StreamlitApp(fig1=example_image_path, fig2=fig2)
    app.run()


