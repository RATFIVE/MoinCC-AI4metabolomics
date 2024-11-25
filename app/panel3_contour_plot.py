import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt


from pathlib import Path
from LoadData import *

load_data = LoadData()


class ContourPlot():
    def __init__(self, df):
        self.df = df

    def data(self):
        df_z = self.df.iloc[:, 1:]
        #z = df_z.values
        z = df_z
        return z
    
    def range(self):
        df_z = self.df.iloc[:, 1:]
        print(df_z)
         #Slider to select z-value range
        z_min, z_max = st.slider(
            "Select the range of values to display in the contour plot",
            min_value=df_z.min().min(),  # Minimum value in the data
            max_value=df_z.max().max(),  # Maximum value in the data
            value=(df_z.min().min(), df_z.max().max())  # Default range
            )
        #z_min, z_max = 10, 10
        
        return z_min, z_max
    #
    
    
    #@st.cache_data
    def plot(self, zmin=10, zmax=20):
        z = self.data()
        #z_min, z_max = self.range()
        #print(z_min, z_max)

        # Extract x, y, z
        x = np.arange(len(z.columns))  # x values (indices of the columns)
        y = np.arange(len(z))          # y values (indices of the rows)
        X, Y = np.meshgrid(x, y)        # Create a grid
        Z = z.values                   # DataFrame values for Z

        # Create a figure
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('white')  # Set the figure background to white
        contour = ax.contourf(X, Y, Z, levels=20, cmap='Blues', vmin=zmin, vmax=zmax)
        #fig.colorbar(contour, label='Value')  # Add a color bar

        # Add labels and title
        #ax.set_title('Contour Plot from DataFrame')
        #ax.set_xlabel('Columns (x)')
        #ax.set_ylabel('Rows (y)')


        #fig = go.Figure()

        # fig.add_trace(
        #     go.Contour(
        #         z=z,
        #         zmin=z_min,
        #         zmax=z_max,
        #         line_smoothing=0,
        #         contours_coloring="heatmap",  # Optional: better visibility
        #         colorbar_title="Intensity",
                
        #         )
        #     )
                # Interaktivität deaktivieren
        #fig.update_layout(hovermode=False)
        
        return fig




# Load data
@st.cache_data
def load_data(file):
    loaddata = LoadData()
    df_list = loaddata.load_data_list(file)
    
    return pd.read_csv(df_list[0])



# Main function to run the app

def main(file_path='/Users/marco/Documents/MoinCC-AI4metabolomics/Data/FA_20240213_2H_yeast_Fumarate-d2_9.csv', zmin=10, zmax=10):
    # Load data
    file = file_path.split('/')[-1]

    df = load_data(file)
    
    fig = ContourPlot(df=df)
    fig = fig.plot(zmin=zmin, zmax=zmax)

    fig.show()

    #st.plotly_chart(fig)
    #st.pyplot(fig)
    
    return fig

if __name__ == "__main__":
    main()
