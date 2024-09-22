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
    
    # Add traces
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='lines'), row=1, col=1)
    fig.update_layout(title='Stacked Lines')
    fig.show()

def main():
    path_list = load_data()
    #print(path_list)
    df = load_df(path_list[1])
    plotly_line(df)
    #df = melt_df(df)
    print(df)
    



if __name__ == '__main__':
    main()