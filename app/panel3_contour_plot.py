import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from classes import LoadData, PeakFinder
from pathlib import Path

# Load data
@st.cache_data
def load_data(file_path):
    loaddata = LoadData()
    df_list = loaddata.load_data(file_path)
    return pd.read_csv(df_list[0])

# Create contour plot
@st.cache_data
def contour_plot(z, zmin, zmax):
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            z=z,
            zmin=zmin,
            zmax=zmax,
            line_smoothing=0,
            contours_coloring="heatmap",  # Optional: better visibility
            colorbar_title="Intensity"
        )
    )
    return fig

# Main function to run the app

def main():
    # Load data
    file = str(Path('FA_20231123_2H Yeast_Fumarate-d2_12 .csv'))
    df = load_data(file)
    
    # Get the values of z directly from the DataFrame (no need to convert to list)
    df_z = df.iloc[:, 1:]  # Assuming data of interest is in all columns except the first
    z = df_z.values  # Avoid converting to list for Plotly compatibility
    
    # Slider to select z-value range
    z_min, z_max = st.slider(
        "Select the range of values to display in the contour plot",
        min_value=df_z.min().min(),  # Minimum value in the data
        max_value=df_z.max().max(),  # Maximum value in the data
        value=(df_z.min().min(), df_z.max().max())  # Default range
    )
    
    # Generate and display the contour plot
    fig = contour_plot(z, zmin=z_min, zmax=z_max)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
