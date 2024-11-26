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
from LoadData import *
from panel3_contour_plot import *
from peak_fitting_v6 import PeakFitting
import matplotlib.pyplot as plt
from Process4Panels import Process4Panels
import plotly.express as px
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from panel1_spectrum_plot import Panel1SpectrumPlot
from panel2_kinetic_plot import KineticPlot
from panel3_contour_plot import ContourPlot





st.set_page_config(layout="wide", page_title="MoinCC - Application", page_icon=":shark:")

meta_fp = os.path.join(os.getcwd(), '..', 'Data', 'Data_description_main.xlsx')
data_fp = os.path.join(os.getcwd(), '..', 'Data', 'FA_20231122_2H_yeast_acetone-d6_3.csv')
referece_fp = os.path.join(os.getcwd(), '..', 'Data', 'FA_20240729_2H_yeast_Reference standard_PBS+Yeast.ser.csv')

class StreamlitApp():
    """
    A Streamlit-based application for visualizing and analyzing metabolite spectra and kinetics.

    This application provides a user-friendly interface for uploading CSV files, visualizing
    plots such as substrate plots, kinetic plots, contour plots, and reference plots, and 
    navigating between panels for different types of analyses.

    Attributes:
        fig1 (str or plotly.graph_objects.Figure): Spectrum plot or its file path.
        fig2 (str or plotly.graph_objects.Figure): Kinetic plot or its file path.
        fig3 (str or plotly.graph_objects.Figure): Contour plot or its file path.
        fig4 (str or plotly.graph_objects.Figure): Reference plot or its file path.
        fig5 (str or plotly.graph_objects.Figure): Additional plot or its file path.
    
    Methods:
        side_bar():
            Creates a sidebar in the Streamlit app with instructions for usage.
        
        header():
            Displays the application header and file uploader, and initializes navigation tabs.

        main_page(main):
            Manages the main content of the application, displaying different panels for analysis.

        about_page(about):
            Displays the "About" section of the application with descriptive text.

        panel1():
            Displays the content for Panel 1, focusing on the substrate plot.

        panel2():
            Displays the content for Panel 2, focusing on the kinetic plot.

        panel3():
            Displays the content for Panel 3, allowing for further exploration.

        panel4():
            Displays the content for Panel 4, offering additional visualizations or data.

        run():
            Executes the Streamlit application, starting from the header and displaying all panels.
    """

    def __init__(self, 
                 fig1=None, 
                 fig2=None,
                 fig3=None,
                 fig4=None
                 ):
        
        self.fig1 = fig1
        self.fig2 = fig2
        self.fig3 = fig3
        self.fig4 = fig4

# /home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/Data/Data_description_main.xlsx
# /home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/Data/FA_20240731_2H_yeast_Fumarate-d2_15_200.ser.csv

# '/Users/marco/Documents/MoinCC-AI4metabolomics/Data/Data_description_main.xlsx'
# '/Users/marco/Documents/MoinCC-AI4metabolomics/Data/FA_20240207_2H_yeast_Fumarate-d2_5.csv'



    def header(self):
        st.markdown("""<h1 style="text-align: center;">MoinCC - Application</h1>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.2, 0.8, 0.2])
        with col1:
            st.divider()

        with col2:
            self.meta_fp = st.text_input('Metadata File Path', meta_fp)
            self.data_fp = st.text_input('Data File Path', data_fp)
            self.referece_fp = st.text_input('Reference File Path', referece_fp)
        
        with col3:
            st.divider()
            if st.button("Start Processing"):
                st.session_state["processing_started"] = True
                try:
                    self.process_data()
                except:
                    st.write('Please select a file.')
               

        main, about = st.tabs(['Main Page', 'Instructions'])

        # Dynamically run main_page only if the button is clicked
        if st.session_state.get("processing_started", True): # Set to false if it should open after pressing the button
            if not st.session_state.get("file_name", self.data_fp):
                st.session_state["file_name"] = self.data_fp
                
            self.main_page(main)
            self.about_page(about)
        
    def main_page(self, main):
        with main:
            st.markdown("### Main Page Content")
            if st.session_state.get("processing_started", True): # Set to false if it should open after pressing the button
                self.panel1()
                self.panel2()
                self.panel3()
            else:
                st.info("Click 'Start Processing' to see the analysis panels.")
    
    def process_data(self):
        #perform peak fitting
        fitter = PeakFitting(self.data_fp, self.meta_fp)
        fitter.fit()
        #Create data for the panels
        processor = Process4Panels(self.data_fp)
        processor.save_sum_spectra()
        processor.save_substrate_individual()
        processor.save_difference()
        processor.save_kinetics()

    def about_page(self, about):
        with about:
            st.markdown(f"""
                        ### Instructions:

                        #### Step 1: 
                        - Select the Metafile Path
                        - Select the Substrate File Path
                        - Select the Reference File Path

                        #### Step 2:
                        - Click Start Processing

                        #### Step 3:
                        ### Substrate Plot
                        - Use Sliders to slect the Frame, to investigate
                        
                        ##### Contour Plot
                        - Use Slider to selct the depth in % 


                        #### Reference Plot
                        - Enjoy

                        
                        
                        """)
        return None

    # -------- Panels with Expanders --------------------
    def panel1(self):
        # read in results from fitting
        sum_fit_fp = Path('output', f'{os.path.basename(self.data_fp)}_output', 'sum_fit.csv')
        sum_fit = pd.read_csv(sum_fit_fp)
        with st.expander("Panel 1 - Substrate Plot", expanded=True):
            # add a slider to select the frame
            st.session_state['time_frame'] = st.slider('Select the frame', min_value=1, max_value=sum_fit.shape[1], value=1)
            st.markdown('# Substrate Plot')
            panel_1_obj = Panel1SpectrumPlot(file_path = self.data_fp)
            raw_plot, lorentz_plot, noise_plot = panel_1_obj.plot(st.session_state['time_frame'])
            st.plotly_chart(raw_plot, use_container_width=True)
            st.plotly_chart(lorentz_plot, use_container_width=True)
            st.plotly_chart(noise_plot, use_container_width=True)

    def panel2(self):
        """ Kinetic Plot"""
        with st.expander("Panel 2 - Kinetic Plot", expanded=True):
            st.markdown('# Kinetic Plot')
            plot = KineticPlot(self.data_fp)
            fig = plot.plot() 
            st.plotly_chart(fig, use_container_width=True)
    
    def panel3(self):
        """Contour Plot"""
        with st.expander("Panel 3 - Contour Plot", expanded=True):
            st.markdown('# Contour Plot')
            panel_3_obj = ContourPlot(self.data_fp)
            # one range slider for both max and min
            zmin_zmax = st.slider('Select Zmin and Zmax', min_value=0.0, max_value=1.0, value=(0.0, 1.0))
            contourplot = panel_3_obj.plot(zmin=zmin_zmax[0], zmax=zmin_zmax[1])
            st.pyplot(contourplot, clear_figure=True)

    
    def run(self):
        self.header()
