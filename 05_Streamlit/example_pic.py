import plotly.express as px
import polars as pl
import numpy as np
import pandas as pd 

def plot_picture(df):
    pic = df.iloc[:, 1:].to_numpy()
    pic = np.transpose(pic)
    
    # flip the picture upside down
    pic = np.flipud(pic)

    # Extract x-axis labels from the first column of the DataFrame
    x_labels = df.iloc[:, 0].to_numpy()

    # extract y-axis labels from the column names of the DataFrame
    y_labels = df.iloc[:, 1:].columns 
    
    # reorder the y-axis labels last to first
    y_labels = y_labels[::-1]


    # Create the heatmap with the specified x-axis labels
    fig = px.imshow(
        pic,
        aspect='auto',
        labels=dict(x="Spectrum", y="Time", color="Intensity"),
        x=x_labels,  # Set the x-axis labels
        y=y_labels,  # Set the y-axis labels 
        # color
        color_continuous_scale='Spectral_r',
    )

    #fig.show()
    return fig


