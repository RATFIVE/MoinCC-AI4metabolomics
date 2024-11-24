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
from panel1_spectrum_plot import main as panel1_main
from panel2_kinetic_plot import KineticPlot
from panel3_contour_plot import main as panel3_main


loaddata = LoadData()



st.set_page_config(layout="wide", page_title="MoinCC - Application", page_icon=":shark:")

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

        panel5():
            Displays the content for Panel 5, for extended analysis or reference.

        run():
            Executes the Streamlit application, starting from the header and displaying all panels.
    """

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
        st.session_state['processing_started'] = False


    def side_bar(self):
        st.sidebar.title('How to')
        
        st.sidebar.markdown(
            """
            1. Upload the .CSV-File 
            2. Use the buttons to navigate panels
            3. Explore the analysis
            """
        )    
# /home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/Data/Data_description_main.xlsx
# /home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/Data/FA_20240731_2H_yeast_Fumarate-d2_15_200.ser.csv

# '/Users/marco/Documents/MoinCC-AI4metabolomics/Data/Data_description_main.xlsx'
# '/Users/marco/Documents/MoinCC-AI4metabolomics/Data/FA _20240215_2H_Yeast_Pyruvate-d3_3.csv'
    def header(self):
        # init se
        st.markdown("""<h1 style="text-align: center;">MoinCC - Application</h1>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([0.2, 0.8, 0.2])
        with col1:
            st.divider()

        with col2:

            self.meta_fp = st.text_input('Metadata File Path', '/Users/marco/Documents/MoinCC-AI4metabolomics/Data/Data_description_main.xlsx')
            self.data_fp = st.text_input('Data File Path', '/Users/marco/Documents/MoinCC-AI4metabolomics/Data/FA_20240207_2H_yeast_Fumarate-d2_5.csv')
            

        with col3:
            st.divider()
            if st.button("Start Processing"):
                st.session_state["processing_started"] = True
        main, about = st.tabs(['Main Page', 'About'])

        # Dynamically run main_page only if the button is clicked
        if st.session_state.get("processing_started", True): # Set to false if it should open after pressing the button
            self.main_page(main)
        self.about_page(about)
        
    def main_page(self, main):
        # perform peak fitting
        #fitter = PeakFitting(self.data_fp, self.meta_fp)
        #fitter.fit()
        # Create data for the panels
        #processor = Process4Panels(self.data_fp)
        #processor.save_sum_spectra()
        #processor.save_substrate_individual()
        #processor.save_difference()
        #processor.save_kinetics()


        with main:
            st.markdown("### Main Page Content")

            if st.session_state.get("processing_started", True): # Set to false if it should open after pressing the button
                self.panel1()
                self.panel2()
                self.panel3()
                self.panel4()
                self.panel5()
            else:
                st.info("Click 'Start Processing' to see the analysis panels.")
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
        # read in results from fitting
        sum_fit_fp = Path('output', f'{os.path.basename(self.data_fp)}_output', 'sum_fit.csv')
        sum_fit = pd.read_csv(sum_fit_fp)
        with st.expander("Panel 1 - Substrate Plot", expanded=True):
            st.markdown('# Substrate Plot')
            raw_plot, lorentz_plot, noise_plot = panel1_main(file_path=self.data_fp, frame=2)
            st.plotly_chart(raw_plot)
            st.plotly_chart(lorentz_plot)
            st.plotly_chart(noise_plot)

        return None

    def panel2(self):
        """ Kinetic Plot"""
        
        with st.expander("Panel 2 - Kinetic Plot", expanded=True):
            st.markdown('# Kinetic Plot')
            plot = KineticPlot(self.data_fp)
            fig = plot.plot() 
            st.plotly_chart(fig)

        return None
    @st.cache_data(experimental_allow_widgets=True)
    def panel3(self):
        """Contour Plot"""
        with st.expander("Panel 3 - Kinetic Plot", expanded=True):
            st.markdown('# Panel 3')
            file = self.data_fp.split('/')[-1]
            df_list = loaddata.load_data_list(file)
            df_z = pd.read_csv(df_list[0])

             #Slider to select z-value range
             
            z_min, z_max = st.slider(
                "Select the range",
                min_value=df_z.min().min(),  # Minimum value in the data
                max_value=df_z.max().max(),  # Maximum value in the data
                value=(df_z.min().min(), df_z.max().max())  # Default range
            )

            self.fig3 = panel3_main(file_path=self.data_fp, zmin=z_min, zmax=z_max)
            st.pyplot(self.fig3)
            
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



'''
if __name__ == '__main__':


    file = str(Path('FA_20231123_2H Yeast_Fumarate-d2_12 .csv'))
    substrates = loaddata.get_substrate_list(file)
    metabolites = loaddata.get_metabolite_list(file)
    df_list = loaddata.load_data(file)
    df = pd.read_csv(df_list[0])
    

   # example_image_path = str(Path(r'/Users/marco/Documents/MoinCC-AI4metabolomics/05_Streamlit/example/FA_20240228_2H_yeast_fumarate-d2_4.csv_output/FA_20240228_2H_yeast_fumarate-d2_4.csv_time_dependence.png'))
    
    fig3 = ContourPlot(df=df)
    fig3 = fig3.plot()
    # Run Streamlit App
    app = StreamlitApp(
                       fig3=fig3)
    app.run()'''


