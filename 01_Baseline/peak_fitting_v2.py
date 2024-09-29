import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
# fitting library
from scipy.optimize import curve_fit
import sys

# Questions:
# How is the concentration calculated? How is the integral calculated?
# How big is the shift? Is it only shifted? Or also skewed?

# get file names
def get_file_names():
    path_list = []
    # data_direc independent of the OS
    data_direc = Path('..','Data')
    # get all filenames which end with .csv
    for file in os.listdir(data_direc):
        if file.endswith('.csv'):
            path_list.append(file)
    return path_list

def containing_string(file_names, string = '', not_string = None):
    # get all filenames which contain the string
    return [file for file in file_names if string in file and (not_string is None or not_string not in file)]

def extract_ppm_all(meta_df, file_name):
    meta_df = meta_df[meta_df['File'].astype(str).str.upper() == str(file_name).upper()]
    if meta_df.shape[0] == 0:
        print(f'No metadata found for {file_name}')
        return [], []
    positions = []
    names = []

    # added substrat like acetone ppm
    print(meta_df['Substrate_chemical shift (ppm)'])
    react_substrat = str(meta_df['Substrate_chemical shift (ppm)'].iloc[0]).split(',')
    for i in range(len(react_substrat)):
        names.append('ReacSubs')
        positions.append(float(react_substrat[i]))
  
    # add metabolite 1
    for i in range(1, 6 ):
        react_metabolite = str(meta_df[f'Metabolite_{i}_ppm'].iloc[0]).split(',')
        if react_metabolite == ['nan']:
            continue
        for j in range(len(react_metabolite)):
            print(react_metabolite)
            names.append(f'Metab{i}')
            positions.append(float(react_metabolite[j]))

    # water ppm
    positions.append(float(meta_df['Water_chemical shift (ppm)'].iloc[0]))
    names.append('Water')

    print(positions)
    print(names)
    return positions, names


def plot_single(x, y_fits, positions, names, file_name, df, output_direc, fit_params):
# create output directory if it does not exist
    if not os.path.exists(output_direc):
        os.makedirs(output_direc)

    for i in range(y_fits.shape[1] - 1):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y_fits[:, i], label='Fit')
        print(df.shape)
        plt.plot(df.iloc[:, 0], df.iloc[:, i + 1], label='Data')
        
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Intensity')
        plt.title(f'NMR Spectrum of {file_name}')
        
        colors = [
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 1.0, 0.0),  # Yellow
    (1.0, 0.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0),  # Cyan
    (0.5, 0.0, 0.5),  # Purple
    (0.5, 0.5, 0.0),  # Olive
    (0.0, 0.5, 0.5),  # Teal
    (0.5, 0.5, 0.5),  # Gray
    (0.75, 0.25, 0.0),  # Brown
    (0.25, 0.75, 0.0),  # Lime Green
    (0.0, 0.25, 0.75),  # Deep Blue
    (0.75, 0.0, 0.25),  # Deep Red
    (0.25, 0.0, 0.75)   # Indigo
]

        # Add vertical lines for the ppm
        for j in range(len(positions)):
            # Add vertical lines for ppm positions
            plt.axvline(x=positions[j], linestyle='--', color=colors[j % len(colors)], label=names[j])
            # Extract fitted parameters for this peak
            shift = fit_params[i, 0] + positions[j]     # Overall shift + position[j]
            gamma = fit_params[i, 1 + j]                # Gamma for this peak
            A = fit_params[i, 1 + len(positions) + j]   # Amplitude for this peak

            # Plot the individual Lorentzian for this peak
            plt.plot(x, lorentzian(x, shift, gamma, A), linestyle='--', color=colors[j % len(colors)], label=f'Peak {names[j]}')

        #no also plot each individual
        # Add legend to avoid repetition
        plt.legend()

        # Save the plot
        plt.savefig(f'{output_direc}/{file_name}_{i}.png')
        plt.close()



def main():
    file_names  = get_file_names()
    #file_names = containing_string(file_names, string='Pyruvate', not_string='ser')

    for file_name in file_names:
        print(f'Processing {file_name}')
        df = pd.read_csv(f'../Data/{file_name}')
        # meta path independet of the OS
        meta_path = Path('..', 'Data', 'Data_description.xlsx')
        # meta_path = '../Data/Data_description.xlsx'
        meta_df = pd.read_excel(meta_path)
        # extract ppm lines and names
        positions, names = extract_ppm_all(meta_df, file_name)
        if len(positions) == 0:
            print(f'No ppm values found for {file_name}')
            continue
        # define bounds for peakfitting
        # width lower bounds
        width_lower_bounds = np.array([0]* len(positions))
        # width upper bounds
        width_upper_bounds = np.array([np.inf]* len(positions))

        # amplitude lower bounds
        amplitude_lower_bounds = np.array([0]* len(positions))
        # amplitude upper bounds
        amplitude_upper_bounds = np.array([np.inf]* len(positions))

        # shift lower bounds
        shift_lower_bounds = np.array([-np.inf])
        # shift upper bounds
        shift_upper_bounds = np.array([np.inf])

        # combine bounds
        lower = np.concatenate([shift_lower_bounds, width_lower_bounds, amplitude_lower_bounds])
        upper = np.concatenate([shift_upper_bounds, width_upper_bounds, amplitude_upper_bounds])
        flattened_bounds = (lower, upper)

        number_peaks = len(positions)
        y_fits = np.zeros((df.shape[0], df.shape[1]))
        # the shift parameter will be shared between all peaks
        fit_params = np.zeros((df.shape[1], number_peaks * 2 + 1))
        for i in range(1, df.shape[1]):
            try:
                print(f'Fitting column {i/df.shape[1]*100:.2f}%')
                # perform fitting
                x = df.iloc[:,0]
                y = df.iloc[:,i]
                popt, pcov = curve_fit(lambda x, *params: grey_spectrum(x, positions, *params),
                                    x, y, p0=[0] + [0.1]*len(positions) + [1000]*len(positions),
                                    maxfev=20000, bounds=flattened_bounds)
                y_fits[:,i-1] = grey_spectrum(x,positions, *popt)
                fit_params[i-1] = popt
            except Exception:
                # what is the error
                import traceback
                traceback.print_exc()
                print('Fitting failed.')
                continue
        output_direc = f'output/2nd_fit/{file_name}_output'
        plot_single(x, y_fits, positions, names, file_name, df, output_direc, fit_params)


def lorentzian(x, shift, gamma, A):
    '''
    x is the datapoint
    x0 is the peak position
    gamma is the width
    A is the amplitude
    '''
    return A * gamma / ((x - shift)**2 + gamma**2)


def grey_spectrum(x, positions,*params):
    '''
    x is the independent variable array
    params should be a flattened list of shift, gamma, and A values
    '''
    n = (len(params) - 1 ) // 2  # The number of peaks
    shift = params[0]            # Single shift parameter
    gamma = params[1:n+1]        # Extract n gamma values
    A = params[n+1:]             # Extract n A values


    y = np.zeros(len(x))
    for i in range(n):
        # Peak position is shared between all peaks
        y += lorentzian(x, shift + positions[i], gamma[i], A[i])
    return y


main()
