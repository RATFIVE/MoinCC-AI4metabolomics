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
from scipy.integrate import trapezoid


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

class PeakDetection:
    def __init__ (self, path):
        self.path = path

        


    def load_df(self, path):
        df = pd.read_csv(path, sep=',', encoding='utf-8')
        df.rename(columns={'Unnamed: 0': 'Chemical_Shift'}, inplace=True)
        return df

    def plotly_line(self):
        
        # load data
        df = self.load_df(self.path)

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

    def peaks_finder(self, y):
        # Finde Peaks in y 
        tresh = self.threshold(y)
        peaks, _ = find_peaks(y, height=tresh)
        return peaks
    
    def threshold(self, y):
        mean = np.mean(y, axis=0)
        std = np.std(y)

        tresh = mean + std
        #print(f'Mean: {mean}')
        return tresh


    def animate(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from scipy.signal import find_peaks

        # load data
        df = self.load_df(self.path)

        peak_tresh = self.threshold(df.iloc[:, 1])

        # Subplots 
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True)
        x = df['Chemical_Shift']
        y = df.iloc[:, 1]
        
            # Initial trace
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='lines'), row=1, col=1)
        
        # Plot peaks to the graph
        fig.add_trace(go.Scatter(x=x[self.peaks_finder(y)], y=y[self.peaks_finder(y)], mode='markers', marker=dict(color='red', size=8), name='Peaks'), row=1, col=1)
        
        # Animation frames
        frames = []
        
        for i in range(1, df.shape[1]):
            y = df.iloc[:, i]
            #print(f'x: {x}')
            #print(f'y: {y}')
            peaks = self.peaks_finder(y)
            frames.append(go.Frame(
                data=[
                    go.Scatter(x=x, y=y, mode='lines'),
                    go.Scatter(x=x[peaks], y=y[peaks], mode='markers', marker=dict(color='red', size=8))
                ],
                name=f'Frame {i}'
            ))
            peak_dict = {'x_values': x[peaks], 'y_values': y[peaks]}
            
        #print(f'Frames: {frames}')
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
        #st.plotly_chart(fig)
        #fig.show()

        return fig

    def intigrate_peaks(self):
        # Load data
        df = self.load_df(self.path)
        x = df['Chemical_Shift']
        print(df.shape)
        
        integrated_y_values = []
        for i in range(1, df.shape[1]):
            y = df.iloc[:, i]
            peaks = self.peaks_finder(y)
            print(f'Peaks: \n{x[peaks].values} \n{y[peaks].values}')

        # return fig
    def bin_and_plot(self, bin_index):
        # Load data
        df = self.load_df(self.path)
        x = df['Chemical_Shift']
        
        binned_y_values = []
        i_values = []

        # Define bins
        num_bins = 10
        bins = np.linspace(x.min(), x.max(), num_bins + 1)

        for i in range(1, df.shape[1]):
            y = df.iloc[:, i]
            peaks = self.peaks_finder(y)
            x_peaks = x[peaks].values
            y_peaks = y[peaks].values

            # Bin the x_peaks and calculate mean y_peaks for each bin
            bin_indices = np.digitize(x_peaks, bins) - 1
            bin_means = [y_peaks[bin_indices == j].mean() if len(y_peaks[bin_indices == j]) > 0 else np.nan for j in range(num_bins)]

            binned_y_values.append(bin_means[bin_index])
            i_values.append(i)
        
        # if bun_index is empty return None
        if np.isnan(binned_y_values).all():
            return None
        else:

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=i_values, 
                                     y=binned_y_values, 
                                     mode='markers+lines', 
                                     name=f'Bin {bin_index}',
                                     line=dict(color='blue', width=2)))

            # Plot the regression line
            poly_coeffs = np.polyfit(i_values, binned_y_values, 7)
            poly_eq = np.poly1d(poly_coeffs)
            fig.add_trace(go.Scatter(x=i_values, 
                                    y=poly_eq(i_values), 
                                    mode='lines', 
                                    name='Regression Line',
                                    line=dict(color='red', width=2)))
        
            fig.update_layout(title=f'Values for Bin {bin_index} over Time at bin range {bins[bin_index]:.2f} to {bins[bin_index + 1]:.2f}', 
                              xaxis_title='Time (i)', yaxis_title='Mean y-values in Bin')
            

            return fig

def main():
    path_list = load_data()

    # make cols for each path in the path_list
    
    try:
        for path in path_list:
            model = PeakDetection(path)
            fig = model.animate()
            st.plotly_chart(fig)

            for bin in range(10):
                try:
                    fig = model.bin_and_plot(bin)
                    st.plotly_chart(fig)
                except:
                    pass


    except Exception as e:
        print(f'Error: {e}')



def main2():
    path_list = load_data()

    model = PeakDetection(path_list[1])
    fig = model.animate()
    st.plotly_chart(fig)

    for bin in range(10):
        try:
            fig = model.bin_and_plot(bin)
            st.plotly_chart(fig)
        except:
            pass



    



if __name__ == '__main__':
    main2()