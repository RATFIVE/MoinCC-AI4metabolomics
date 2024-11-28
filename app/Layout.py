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
import matplotlib.pyplot as plt
from Process4Panels import Process4Panels
import plotly.express as px
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from panel1_spectrum_plot import Panel1SpectrumPlot
from panel2_kinetic_plot import KineticPlot
from panel3_contour_plot import ContourPlot
from panel5_reference_plot import Reference
#from peak_fitting_v7 import PeakFitting



st.set_page_config(layout="wide", page_title="MoinCC - Application", page_icon=":shark:")
# Custom CSS to change font size of buttons and other widgets


meta_fp = os.path.join(os.getcwd(), '..', 'Data', 'Data_description_main.xlsx')
data_fp = os.path.join(os.getcwd(), '..', 'Data', 'FA_20240517_2H_yeast_Nicotinamide-d4 _6.csv')
# FA_20240207_2H_yeast_Fumarate-d2_5.csv
# FA_20231122_2H_yeast_acetone-d6_3.csv
reference_fp = os.path.join(os.getcwd(), '..', 'Data', 'FA_20240806_2H_yeast_Reference_standard_PBS.ser.csv')

# Function to open a file dialog and get the file path
def select_file(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]):
    """load data path

    Returns:
        dir_path (str): directory path
    """
    root = Tk()
    file_path = filedialog.askopenfilename(filetypes=filetypes)
    #dir_path = r'/Users/marco/Documents/UKSH/Data/Banzhaf_Marco_1996-09-03_B.MA._2/2024-04-11'
    root.destroy()
    if file_path == '':
        return None
    return file_path

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
            
            # Selector for choosing the model
            options = ["Model 1", "Model 2"]
            selected_option = st.selectbox("Choose the Model:", options)

            # Dynamically load and store the class in session_state
            if selected_option == "Model 1":
                
                #if 'Model 1' in st.session_state:
                st.session_state['Model 1'] = True  # Store the class in session state
                st.session_state['Model 2'] = False 
            
            elif selected_option == "Model 2":
                
                #if 'Model 2' in st.session_state:
                st.session_state['Model 1'] = False  # Store the class in session state
                st.session_state['Model 2'] = True


            sub_col1, sub_col2 = st.columns([0.30, 0.70])

            # Select the Metadata as xml
            with sub_col1:
                st.markdown('**Step 1: Select the Metadata as .xlsx**')
                select_meta_button = st.button(label='Select Metafile')
                if select_meta_button:
                    self.meta_fp = select_file(filetypes=[("XLSX files", "*.xlsx")])
                    st.session_state["meta_file"] = self.meta_fp
                
            # Select the Spectrum as csv 
            with sub_col1:
                st.markdown('**Step2: Select Spectrum as .csv**')
                select_file_button = st.button(label='Select Spectrum', )
                if select_file_button:
                    self.data_fp = select_file(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
                    st.session_state["file_name"] = self.data_fp

            # Select the Reference file as 
            with sub_col1:
                st.markdown('**Step3: Select the Reference File as .csv**')
                select_reference_button = st.button(label='Select Reference')
                if select_reference_button:
                    self.reference_fp = select_file(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
                    st.session_state['reference_file'] = self.reference_fp
                    

        # Session State for Model import
        if 'Model 1' in st.session_state:
            from peak_fitting_v7 import PeakFitting
        
        if 'Model 2' in st.session_state:
            from peak_fitting_v6 import PeakFitting

        # Session State for files
        if 'meta_file' in st.session_state:
            self.meta_fp = st.session_state['meta_file']
            with sub_col2:
                    st.info(self.meta_fp)
        else:
            st.warning("No meta file selected or key does not exist.")

        if 'reference_file' in st.session_state:
            self.reference_fp = st.session_state['reference_file']
            with sub_col2:
                st.info(self.reference_fp)
            
        else:
            st.warning("No reference file selected or key does not exist.")

        if "file_name" in st.session_state:
            self.data_fp = st.session_state["file_name"]
            with sub_col2:
                    st.info(self.data_fp)
        else:
            st.warning("No substrate file selected or key does not exist.")
        with col2:
            process_col1, process_col2, process_col3 = st.columns([2, 1, 1])  # 1:2:1 ratio
        with process_col1:
            st.markdown('**Step4: Press Start Processing**')
        with process_col2:
            if st.button("Start Processing"):
                    st.write(f"Processing data from: {self.data_fp}")
                    st.session_state["processing_started"] = True
                    self.process_data(PeakFitting)
                    
        with col3:
            st.divider()


        # Slit into Main and Instruction Tabs
        main, about = st.tabs(['Main Page', 'Instructions'])


        # Dynamically run main_page only if the button is clicked
        if 'processing_started' in st.session_state:
            if st.session_state["processing_started"]: # Set to false if it should open after pressing the button
                if st.session_state["file_name"] is not None:
                    self.main_page(main)
        self.about_page(about)
        
    def main_page(self, main):
        with main:
            st.markdown("#### Main Page Content")
            if st.session_state.get("processing_started", True): # Set to false if it should open after pressing the button
                self.panel1()
                self.panel2()
                self.panel3()  
                self.panel4()
            else:
                st.info("Click 'Start Processing' to see the analysis panels.")
    
    def process_data(self, PeakFitting):
        #perform peak fitting
        fitter = PeakFitting(self.data_fp, self.meta_fp)
        fitter.fit()
        #Create data for the panels
        processor = Process4Panels(self.data_fp)
        processor.save_sum_spectra()
        processor.save_substrate_individual()
        processor.save_difference()
        processor.save_kinetics()
    
    def process_plots(self):
        self.panel_1_obj = Panel1SpectrumPlot(file_path = self.data_fp)
        self.panel_2_obj = KineticPlot(self.data_fp)
        self.panel_3_obj = ContourPlot(self.data_fp)
        self.panel_4_obj = Reference(fp_file = self.reference_fp, fp_meta = self.meta_fp)

    def about_page(self, about):
        with about:
            # Text which will be shown in the Instruction 
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
        try:
            sum_fit_fp = Path('output', f'{os.path.basename(self.data_fp)}_output', 'sum_fit.csv')
        
            sum_fit = pd.read_csv(sum_fit_fp)
        except:
            # Show this text message if there the Output Data is no available.
            st.markdown("""
                        <span style="color:red; font-size:72px;">Please Press 'Start Processing !'</span>
                        """, unsafe_allow_html=True)

        with st.expander("Panel 1 - Substrate Plot", expanded=True):
            # add a slider to select the frame
            st.session_state['time_frame'] = st.slider('Select the frame', min_value=1, max_value=sum_fit.shape[1], value=1)
            st.markdown('# Substrate Plot')
            st.write('Inside Panel 1')
            panel_1_obj = Panel1SpectrumPlot(file_path = self.data_fp)
            one_plot = panel_1_obj.plot(st.session_state['time_frame'])
            st.plotly_chart(one_plot, use_container_width=True)

            
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

    def panel4(self):
        """Reference Plot"""
        with st.expander("Panel 4 - Reference", expanded=True):
            st.markdown('# Reference')
            Reference_obj = Reference(fp_file = self.reference_fp, fp_meta = self.meta_fp)
            i = st.slider('Select the frame for water reference', min_value=1, max_value= Reference_obj.data.shape[1], value=1) #max_value ist falsch, sessionstate?
            reference_plot = Reference_obj.plot(i = i)
            st.pyplot(reference_plot)


    def run(self):
        self.header()
