import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from LoadData import *
#from Layout import StreamlitApp
import streamlit as st
import plotly.io as pio
import plotly.express as px
from plotly.colors import sequential
from plotly.subplots import make_subplots



class Panel1SpectrumPlot():
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.plot_dir = Path('output', self.file_name + '_output', 'plots')
        self.spectrum_pdf = Path(self.plot_dir, f'Spectrum_{self.file_name}')
        self.noise_pdf = Path(self.plot_dir, f'Noise{self.file_name}')
        self.fitted_pdf = Path(self.plot_dir, f'Fitted_{self.file_name}')
        self.colors = px.colors.qualitative.Plotly
        self.template = 'plotly_white' 

        # Ensure the plot directory exists 
        os.makedirs(self.plot_dir, exist_ok=True)
        
        
        self.data = pd.read_csv(file_path, header = 0)
        self.sum_data = pd.read_csv(Path('output', f'{self.file_name}_output', 'sum_fit.csv'), header = 0)
        self.differences = pd.read_csv(Path('output', f'{self.file_name}_output', 'differences.csv'), header = 0)
        
        # Read in and sort files numerically by the number in their names
        ind_file_names = sorted(
            os.listdir(Path('output', f'{self.file_name}_output', 'substance_fits')),
            key=lambda x: int(x.split('_fit_')[1].split('.csv')[0])  # Extract number for sorting
        )

        # Load the CSV files in the sorted order
        self.individual_fits = [
            pd.read_csv(Path('output', f'{self.file_name}_output', 'substance_fits', f), header=0)
            for f in ind_file_names
        ]

    def plot(self, frame):
        # for consistent y axis
        self.min_y = self.data.iloc[:,1:].min().min() + self.data.iloc[:,1:].min().min()
        self.max_y = self.data.iloc[:,1:].max().max()
        # for constant x axis
        self.min_x = self.data.iloc[:, 0].min()
        self.max_x = self.data.iloc[:, 0].max()
        
        fig1 = self.plot_raw(frame)
        fig2 = self.plot_sum_fit(frame)
        fig3 = self.plot_diff(frame)

        fig = make_subplots(rows=1, cols=1, subplot_titles=(" ", " ", " "))
    
        # Add traces from fig_raw to the first subplot
        for trace in fig1.data:
            fig.add_trace(trace, row=1, col=1)
        
        # Add traces from fig_diff to the second subplot
        for trace in fig2.data:
            fig.add_trace(trace, row=1, col=1)
        
        # Add traces from fig_sm_fit to the third subplot
        for trace in fig3.data:
            fig.add_trace(trace, row=1, col=1)

        fig.update_layout(
            title=dict(
                text='Spectrum',
                font=dict(size=30)  # Font size for the title
            ),
            xaxis_title='Chemical shift [ppm]',
            yaxis_title='Intensity',
            showlegend=True,
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            yaxis=dict(
                range=[self.min_y, self.max_y],
                titlefont=dict(size=24),          # Font size for y-axis title
                tickfont=dict(size=18)            # Font size for y-axis ticks
            ),
            xaxis=dict(
                range=[self.max_x, self.min_x],
                dtick=0.5,
                titlefont=dict(size=24),          # Font size for x-axis title
                tickfont=dict(size=18)            # Font size for x-axis ticks
            )
                          # To Change direction of x axis from low to high 
        )

        return fig

    def plot_raw(self, frame):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.iloc[:,0][::-1],       # To Change direction of x axis from low to high 
                                 y=self.data.iloc[:,frame][::-1],   # To Change direction of x axis from low to high 
                                 mode='lines', name='Raw',
                                 line=dict(color=self.colors[0])
                                 ))
        
        fig.update_layout(
            title='Spectrum',
            xaxis_title='Chemical shift [ppm]',
            yaxis_title='Intensity',
            #legendgroup='Group 1',
            #showlegend=False,
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            yaxis=dict(range=[self.min_y, self.max_y]),
            xaxis=dict(range=[self.max_x, self.min_x], dtick=0.5),              # To Change direction of x axis from low to high 
            template=self.template
            )    
        # Save the fig as pdf
        self.save_fig(fig, self.spectrum_pdf)  
        return fig
    
    def plot_diff(self, frame):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.differences.iloc[:,0][::-1],        # To Change direction of x axis from low to high 
                                 y=self.differences.iloc[:,frame][::-1],    # To Change direction of x axis from low to high 
                                 mode='lines', name='Diff',
                                 line=dict(color=self.colors[-1])
                                 ))
        fig.update_layout(
            title='Noise',
            xaxis_title='Chemical shift [ppm]',
            yaxis_title='Intensity',
            showlegend=True,
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            yaxis=dict(range=[self.differences.iloc[:,frame].min()*20, self.max_y]),
            xaxis=dict(range=[self.max_x, self.min_x], dtick=0.5),                     # To Change direction of x axis from low to high 
            template=self.template
            )

        # Save the fig as pdf
        self.save_fig(fig, self.noise_pdf) 
        return fig

    def plot_sum_fit(self, frame):
        frame_data = self.individual_fits[frame -1 ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.sum_data.iloc[:,0][::-1],           # To Change direction of x axis from low to high 
                                 y=self.sum_data.iloc[:,frame][::-1],       # To Change direction of x axis from low to high 
                                 mode='lines+markers', 
                                 name='Sum Fit',
                                 line=dict(color=self.colors[-1])
                                 ))
        colors = px.colors.qualitative.Plotly
        for i in range(1, len(frame_data.columns)):
            color = self.colors[(i - 1) % len(colors)]
            fig.add_trace(go.Scatter(x=self.sum_data.iloc[:,0], 
                                     y=frame_data.iloc[:,i], 
                                     mode='lines', 
                                     name=frame_data.columns[i],
                                     #legendgroup='Group4'
                                     line=dict(color=color)
                                     ))

            
        fig.update_layout(
            title='Fitted Lorenzian Curves',
            xaxis_title='Chemical shift [ppm]',
            yaxis_title='Intensity',
            showlegend=True,
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            yaxis=dict(range=[self.min_y, self.max_y]),
            xaxis=dict(range=[self.max_x, self.min_x], dtick=0.5),                      # To Change direction of x axis from low to high 
            template=self.template
            )
        # Save the fig as pdf
        self.save_fig(fig, self.fitted_pdf)
        return fig
    
    def save_fig(self, fig, name):
        pass
        #pio.write_image(fig, f'{name}.pdf', format='pdf', engine='kaleido', width=1200, height=800)
        #pio.write_image(fig, f'{name}.png', format='png', engine='kaleido') 
        


   