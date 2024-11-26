import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from LoadData import *
#from Layout import StreamlitApp
import streamlit as st



class Panel1SpectrumPlot():
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        
        self.data = pd.read_csv(file_path, header = 0)
        self.sum_data = pd.read_csv(Path('output', f'{self.file_name}_output', 'sum_fit.csv'), header = 0)
        self.differences = pd.read_csv(Path('output', f'{self.file_name}_output', 'differences.csv'), header = 0)
        
        # Read in and sort files numerically by the number in their names
        ind_file_names = sorted(
            os.listdir(Path('output', f'{self.file_name}_output', 'substance_fits')),
            key=lambda x: int(x.split('_fit')[1].split('.csv')[0])  # Extract number for sorting
        )

        # Load the CSV files in the sorted order
        self.individual_fits = [
            pd.read_csv(Path('output', f'{self.file_name}_output', 'substance_fits', f), header=0)
            for f in ind_file_names
        ]

    def plot(self, frame):
        # for consistent y axis
        self.min_y = self.data.iloc[:,1:].min().min()
        self.max_y = self.data.iloc[:,1:].max().max()
        # for constant x axis
        self.min_x = self.data.iloc[:, 0].min()
        self.max_x = self.data.iloc[:, 0].max()
        
        fig1 = self.plot_raw(frame)
        fig2 = self.plot_sum_fit(frame)
        fig3 = self.plot_diff(frame)
        return (fig1, fig2, fig3)

    def plot_raw(self, frame):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.iloc[:,0][::-1],       # To Change direction of x axis from low to high 
                                 y=self.data.iloc[:,frame][::-1],   # To Change direction of x axis from low to high 
                                 mode='lines', name='Raw'))
        
        fig.update_layout(
            title='Spectrum',
            xaxis_title='Chemical Shift [ppm]',
            yaxis_title='Intensity',
            showlegend=False,
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            yaxis=dict(range=[self.min_y, self.max_y]),
            xaxis=dict(range=[self.max_x, self.min_x], dtick=0.5)              # To Change direction of x axis from low to high 
        )        
        return fig
    
    def plot_diff(self, frame):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.differences.iloc[:,0][::-1],        # To Change direction of x axis from low to high 
                                 y=self.differences.iloc[:,frame][::-1],    # To Change direction of x axis from low to high 
                                 mode='lines', name='Diff'))
        fig.update_layout(
            title='Noise',
            xaxis_title='Chemical Shift [ppm]',
            yaxis_title='Intensity',
            showlegend=True,
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            yaxis=dict(range=[self.differences.iloc[:,frame].min()*20, self.max_y]),
            xaxis=dict(range=[self.max_x, self.min_x], dtick=0.5)                      # To Change direction of x axis from low to high 
        )
        return fig

    def plot_sum_fit(self, frame):
        frame_data = self.individual_fits[frame -1 ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.sum_data.iloc[:,0][::-1],           # To Change direction of x axis from low to high 
                                 y=self.sum_data.iloc[:,frame][::-1],       # To Change direction of x axis from low to high 
                                 mode='lines', name='Sum Fit'))
        for i in range(1, len(frame_data.columns)):
            fig.add_trace(go.Scatter(x=self.sum_data.iloc[:,0], y=frame_data.iloc[:,i], mode='lines', name=frame_data.columns[i]))

            
        fig.update_layout(
            title='Sum Fit',
            xaxis_title='Chemical Shift [ppm]',
            yaxis_title='Intensity',
            showlegend=True,
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            yaxis=dict(range=[self.min_y, self.max_y]),
            xaxis=dict(range=[self.max_x, self.min_x], dtick=0.5)                      # To Change direction of x axis from low to high 
        )
        return fig