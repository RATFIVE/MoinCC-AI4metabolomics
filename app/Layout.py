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
        with col1:
            st.divider()

        with col2:
            st.file_uploader("Upload the Metabolite Spectrum CSV")

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
        with main:
            st.markdown("### Main Page Content")
            # Show panels only if processing is started
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
        
        with st.expander("Panel 1 - Substrate Plot", expanded=True):
            st.markdown('# Substrate Plot')
            st.write("Content for Panel 1.")

        return None

    def panel2(self):
        """ Kinetic Plot"""
        
        with st.expander("Panel 2 - Kinetic Plot", expanded=True):
            st.markdown('# Kinetic Plot')
            st.image(self.fig2, caption="Your Image Caption", use_column_width=True)

        return None

    def panel3(self):
        """Contour Plot"""
        with st.expander("Panel 3 - Kinetic Plot", expanded=True):
            st.markdown('# Panel 3')
            st.slider
            #st.plotly_chart(self.fig3)
            
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
    

   # example_image_path = str(Path(r'/Users/marco/Documents/MoinCC-AI4metabolomics/05_Streamlit/example/FA_20240228_2H_yeast_fumarate-d2_4.csv_output/FA_20240228_2H_yeast_fumarate-d2_4.csv_time_dependence.png'))
    
    fig3 = ContourPlot(df=df)
    fig3 = fig3.plot()
    # Run Streamlit App
    app = StreamlitApp(
                       fig3=fig3)
    app.run()


