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
import spm1d as spm


# ----- Settings -----------------------------------------------------------
# set streamlit layout to wide
st.set_page_config(layout="wide")

# Options to display all columns
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

def load_data():
    path_list = []
    cwd = Path(os.getcwd())
    print(f'Working Dir: {cwd}')
    data_path = os.path.join(cwd.parent, 'Data')

    print(f'Path: {data_path}')
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                path_list.append(file_path)
    return path_list

# ----- Class for Peak Detection -----------------------------------------------------------
class PeakDetection:
    def __init__ (self, path):
        self.path = path

        self.mean_shift = None
        self.std_shift = None

# --------------------------------------------------------------------------------------------

    def load_df(self, path):
        df = pd.read_csv(path, sep=',', encoding='utf-8')
        df.rename(columns={'Unnamed: 0': 'Chemical_Shift'}, inplace=True)
        return df
    
# --------------------------------------------------------------------------------------------

    def plotly_line(self):

        """Single Line Plot of the Spectra"""
        
        
        # load data
        df = self.load_df(self.path)

        # subplots 
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True)
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        # finde peaks in y 
        peak_tresh = 20000
        peaks, _ = find_peaks(df.iloc[:, 1], height=peak_tresh)
        #print(f'Peaks: {peaks}')
        # Add traces
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='lines'), row=1, col=1)
        # plot peaks to the graph
        fig.add_trace(go.Scatter(x=x[peaks], y=y[peaks], mode='markers', marker=dict(color='red', size=8), name='Peaks'), row=1, col=1)
        fig.update_layout(title='Stacked Lines')
        fig.show()

# --------------------------------------------------------------------------------------------

    def peaks_finder(self, y):
        """Find Peaks in the Spectra

        Args:
            y (float): y values of the spectra

        Returns:
            [int]: Index of the peaks
        """
        # Finde Peaks in y 
        tresh = self.threshold(y)
        peaks, _ = find_peaks(y, height=tresh)
        return peaks
    
# --------------------------------------------------------------------------------------------

    def threshold(self, y):
        mean = np.mean(y, axis=0)
        std = np.std(y)

        tresh = mean + std
        #print(f'Mean: {mean}')
        return tresh

# --------------------------------------------------------------------------------------------
    def animate(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from scipy.signal import find_peaks

        # load data
        df = self.load_df(self.path)
        #df = self.shift_correction()

        peak_tresh = self.threshold(df.iloc[:, 1])
        smoothing = 30
        # Subplots 
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True)
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        y1 = spm.util.smooth(y, fwhm=smoothing)
        
            # Initial trace
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='lines'), row=1, col=1)
        # add smoothed line
        fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Smoothed Line', line=dict(color='orange')), row=1, col=1)
        
        # Plot peaks to the graph
        fig.add_trace(go.Scatter(x=x[self.peaks_finder(y)], y=y[self.peaks_finder(y)], mode='markers', marker=dict(color='red', size=8), name='Peaks'), row=1, col=1)
        
        # Animation frames
        frames = []
        
        for i in range(1, df.shape[1]):
            y = df.iloc[:, i]
            y1 = spm.util.smooth(y, fwhm=smoothing)
            #print(f'x: {x}')
            #print(f'y: {y}')
            peaks = self.peaks_finder(y)
            frames.append(go.Frame(
                data=[
                    go.Scatter(x=x, y=y, mode='lines'),
                    go.Scatter(x=x, y=y1, mode='lines'),
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
    
# --------------------------------------------------------------------------------------------

    def intigrate_peaks(self):
        # Load data
        df = self.load_df(self.path)
        
        #print(df.shape)
        
        integrated_y_values = []
        for i in range(1, df.shape[1]):
            y = df.iloc[:, i]
            peaks = self.peaks_finder(y)
            #print(f'Peaks: \n{x[peaks].values} \n{y[peaks].values}')

# --------------------------------------------------------------------------------------------
    def calculate_peak_deviation(self):
        
        x_ref = 4.7 # Reference value of Water
        # Load data
        df = self.load_df(self.path)
        
        x = df.iloc[:, 0]

        deviation_mean_list = []
        for i in range(1, df.shape[1]):
            y = df.iloc[:, i]
            peaks = self.peaks_finder(df.iloc[:, i])
            #print(f'x Peaks: {x[peaks].values} \ny Peaks: {y[peaks].values}')
            ref_range = 0.1
            x_upper = x_ref + ref_range
            x_lower = x_ref - ref_range
            deviation_list = []
            for peak in x[peaks].values:
                if peak >= x_lower and peak <= x_upper:
                    print(f'Peak: {peak} is within the range of {x_lower} and {x_upper}')

                    # calculate the deviation
                    deviation = peak - x_ref
                    deviation_list.append(deviation)
                else:
                    print(f'Peak: {peak} is not within the range of {x_lower} and {x_upper}')
                    pass

            if deviation_list:
                deviation_mean = np.mean(deviation_list)
                deviation_mean_list.append(deviation_mean)
            else:
                print(f'No peaks within the range for column {i}')

        if deviation_mean_list:
            self.mean_shift = np.mean(deviation_mean_list)
            self.std_shift = np.std(deviation_mean_list)
            
            #print(f'Mean Shift: {self.mean_shift}')
            #print(f'Standard Deviation Shift: {self.std_shift}')
        else:
            #print('No deviations calculated.')
            self.mean_shift = None
            self.std_shift = None

        return self.mean_shift, self.std_shift
    
# --------------------------------------------------------------------------------------------

    def shift_correction(self):
        # Load data
        df = self.load_df(self.path)


        x = df.iloc[:, 0]


        calculated_shift = self.calculate_peak_deviation()
        for i in range(1, df.shape[1]):
            y = df.iloc[:, i]
            df.iloc[:, i] = y - self.mean_shift

        return df

# --------------------------------------------------------------------------------------------
   
    def bin_and_plot(self, bin_index):
        # Load data
        df = self.load_df(self.path)
        #df = self.shift_correction()

        x = df.iloc[:, 0]
            
        

        
        
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
# --------------------------------------------------------------------------------------------

def main():
    path_list = load_data()

    # make cols for each path in the path_list
    col1, col2 = st.columns(2)
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


    except:
        #print(f'Error: {e}')
        pass



def main2():

    col1, col2 = st.columns(2)
    path_list = load_data()

    model = PeakDetection(path_list[1])
    model.calculate_peak_deviation()
    fig = model.animate()
    with col1:
        st.plotly_chart(fig)

    for bin in range(10):
        try:
            fig = model.bin_and_plot(bin)
            with col2:
                st.plotly_chart(fig)
        except:
            pass


def main3():
    means_list = []
    path_list = load_data()
    error_list = []
    try:
        for path in path_list:
            model = PeakDetection(path)
            mean, std = model.calculate_peak_deviation()
            means_list.append(mean)
    except Exception as e:

        error_list.append(path)
        print(f'Error: {e}')

    filtered_means = [m for m in means_list if m is not None]
    means = np.array(filtered_means)
    mean = np.mean(means)
    std = np.std(means)
    print(f'Mean: {mean} \nStd: {std}')
    print(f'Error List: {error_list}')


if __name__ == '__main__':
    main()