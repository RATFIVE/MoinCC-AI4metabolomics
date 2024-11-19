import pandas as pd
import plotly.graph_objects as go
import streamlit as st


from pathlib import Path
from LoadData import *

load_data = LoadData()


class ContourPlot():
    def __init__(self, df):
        self.df = df

    def data(self):
        df_z = self.df.iloc[:, 1:]
        z = df_z.values
        return z
    
    def range(self):
        df_z = self.df.iloc[:, 1:]
         # Slider to select z-value range
        z_min, z_max = st.slider(
            "Select the range of values to display in the contour plot",
            min_value=df_z.min().min(),  # Minimum value in the data
            max_value=df_z.max().max(),  # Maximum value in the data
            value=(df_z.min().min(), df_z.max().max())  # Default range
            )
        
        return z_min, z_max
    #
    

    
    def plot(self):
        z = self.data()
        z_min, z_min = self.range()

        fig = go.Figure()

        fig.add_trace(
            go.Contour(
                z=z,
                zmin=z_min,
                zmax=z_min,
                line_smoothing=0,
                contours_coloring="heatmap",  # Optional: better visibility
                colorbar_title="Intensity")
            )
        
        return fig




# Load data
@st.cache_data
def load_data(file_path):
    loaddata = LoadData()
    df_list = loaddata.load_data(file_path)
    return pd.read_csv(df_list[0])



# Main function to run the app

def main():
    # Load data
    file = str(Path('FA_20231123_2H Yeast_Fumarate-d2_12 .csv'))
    df = load_data(file)
    
    fig = ContourPlot(df=df)
    fig = fig.plot()

    st.plotly_chart(fig)
    
    return fig

if __name__ == "__main__":
    main()
