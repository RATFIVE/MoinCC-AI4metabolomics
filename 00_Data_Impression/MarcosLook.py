import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import shutil
import time
from pathlib import Path
from plotly.subplots import make_subplots
from scipy.signal import find_peaks


# Options to display all columns
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

def load_data():
    path_list = []
    cwd = Path(os.getcwd())
    print(f'Working Dir: {cwd}')

    print(f'Path: {cwd.parent}')
    for root, dirs, files in os.walk(cwd.parent):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                path_list.append(file_path)
    return path_list


def load_df(path):
    df = pd.read_csv(path, sep=',', encoding='utf-8')
    df.rename(columns={'Unnamed: 0': 'Chemical_Shift'}, inplace=True)
    return df

def df_description(df):
    print(f'Dataframe Shape: {df.shape}')
    print(f'DataFrame Head: \n{df.head()}')
    return None

def melt_df(df):
    # Melt the dataframe (Polars)
    melted_df = df.melt(id_vars=[df.columns[0]], value_vars=df.columns[1:])
    #print(melted_df)
    #Rename the columns for better understanding
    melted_df = melted_df.rename({
        df.columns[0]: 'Chemical Shift (ppm)',  # First column
        'variable': 'Time',  # The original column names (used as time points)
        'value': 'Intensity'  # The melted values (intensities)
    })
    return melted_df



def plotly_line(df):

    # subplots 
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True)
    x = df['Chemical_Shift']
    y = df.iloc[:, 1]
    # finde peaks in y 
    peak_tresh = 20000
    peaks, _ = find_peaks(df.iloc[:, 1], height=peak_tresh)
    print(f'Peaks: {peaks}')
    # Add traces
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='lines'), row=1, col=1)
    # plot peaks to the graph
    fig.add_trace(go.Scatter(x=x[peaks], y=y[peaks], mode='markers', marker=dict(color='red', size=8), name='Peaks'), row=1, col=1)
    fig.update_layout(title='Stacked Lines')
    fig.show()

def peaks_finder(y, tresh):
    # Finde Peaks in y 
    tresh = 20000
    peaks, _ = find_peaks(y, height=tresh)
    return peaks


def animate(df):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.signal import find_peaks
    peak_tresh = 20000

    # Subplots 
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True)
    x = df['Chemical_Shift']
    y = df.iloc[:, 1]
    
        # Initial trace
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='lines'), row=1, col=1)
    
    # Plot peaks to the graph
    fig.add_trace(go.Scatter(x=x[peaks_finder(y, tresh=peak_tresh)], y=y[peaks_finder(y, tresh=peak_tresh)], mode='markers', marker=dict(color='red', size=8), name='Peaks'), row=1, col=1)
    
    
    # Animation frames
    frames = []

    
    for i in range(1, df.shape[1]):
        y = df.iloc[:, i]
        #print(f'x: {x}')
        #print(f'y: {y}')
        peaks = peaks_finder(y, tresh=peak_tresh)
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x, y=y, mode='lines'),
                go.Scatter(x=x[peaks], y=y[peaks], mode='markers', marker=dict(color='red', size=8))
            ],
            name=f'Frame {i}'
        ))
    print(f'Frames: {frames}')
    # Add frames to the figure
    fig.frames = frames
    
    # Update layout with animation settings
    fig.update_layout(
        title='Stacked Lines with Animation',
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }]
    )
    
    fig.show()

def main():
    path_list = load_data()
    #print(path_list)
    df = load_df(path_list[1])
    #plotly_line(df)
    animate(df)
    #df = melt_df(df)
    print(df)
    



if __name__ == '__main__':
    main()