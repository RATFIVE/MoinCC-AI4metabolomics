import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
# fitting library
from scipy.optimize import curve_fit

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

def containing_string(file_names, string, not_string = None):
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


def plot_single(x, y_fits, positions, names, file_name, df, output_direc):
    for i in range(y_fits.shape[1] - 1):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y_fits[:, i], label='Fit')
        print(df.shape)
        plt.plot(df.iloc[:, 0], df.iloc[:, i + 1], label='Data')
        
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Intensity')
        plt.title(f'NMR Spectrum of {file_name}')
        
        colors = ['r', 'g', 'b', 'y', 'm', 'c']
        # Add vertical lines for the ppm
        for j in range(len(positions)):
            plt.axvline(x=positions[j], linestyle='--', label=names[j], color=colors[j % len(colors)])
        
        # Add legend to avoid repetition
        plt.legend()
        
        # create output directory if it does not exist
        if not os.path.exists(output_direc):
            os.makedirs(output_direc)

        # Save the plot
        plt.savefig(f'{output_direc}/{file_name}_{i}.png')
        plt.close()

def plot_time_dependece(y_fits, positions, names, popt, output_direc):
    # the area of the indivudal peaks is the concentration with some kind of linear transformation on top.
    area_per_peak = np.zeros((y_fits.shape[1], len(positions)))

    for i in range(y_fits.shape[1]):
        for j in range(len(positions)):
            area_per_peak[i, j] = popt[i,len(positions) + j] * popt[i, len(positions)*2 + j]


    time_points = range(y_fits.shape[1])

    # scatter plot of time evolution of the peak areas
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    for i in range(len(positions)):
        plt.scatter(time_points, area_per_peak[:, i], label=names[i], color=colors[i % len(colors)])

    plt.xlabel('Time')
    plt.ylabel('Area')
    plt.title('Time evolution of peak areas')
    plt.ylim((-0, 50))
    plt.legend()

    #create output directory if it does not exist
    if not os.path.exists(output_direc):
        os.makedirs(output_direc)

    plt.savefig(f'{output_direc}/Time_evolution_areas.png')
    plt.close()    

def correct_starting_point(df, positions, names):
    # Sum up all intensities
    sum_intensities = df.iloc[:, 1:].sum(axis=1) / df.shape[1]
    plt.plot(df.iloc[:, 0], sum_intensities)

    change_positions = np.linspace(-0.3, 0.3, 10)
    errors = np.zeros(len(change_positions))
    parameters = np.zeros((len(change_positions), len(positions) * 3))

    # Fitting all parameters in a range of -0.1 to 0.1
    for i in range(len(change_positions)):
        shift = change_positions[i]
        
        # lower bounds ppm position
        ppm_lower_bounds = [np.array(positions) - 0.3]
        # upper bounds ppm position
        ppm_upper_bounds = [np.array(positions) + 0.3]

        # width lower bounds
        width_lower_bounds = np.array([0]* len(positions))
        # width upper bounds
        width_upper_bounds = np.array([np.inf]* len(positions))

        # amplitude lower bounds
        amplitude_lower_bounds = np.array([0]* len(positions))
        # amplitude upper bounds
        amplitude_upper_bounds = np.array([np.inf]* len(positions))

        # Concatenate all arrays
        all_bounds = np.concatenate((
            np.concatenate(ppm_lower_bounds),
            width_lower_bounds,
            amplitude_lower_bounds,
            np.concatenate(ppm_upper_bounds),
            width_upper_bounds,
            amplitude_upper_bounds
        ))

        # Flatten the output
        flattened_bounds = all_bounds.flatten()

        # reshape for the curve fitting
        flattened_bounds = np.reshape(flattened_bounds, (2, -1))
        # Print the result
        print(flattened_bounds)
        print(np.array([positions + shift, [1e-2] * len(positions), [10] * len(positions)]).flatten())

        # Perform curve fitting
        try:
            popt, pcov = curve_fit(
                grey_spectrum, 
                df.iloc[:, 0], 
                sum_intensities,
                p0=np.array([positions + shift, [0.5e-1] * len(positions), [100] * len(positions)]).flatten(), 
                maxfev=1000, 
                bounds=flattened_bounds
            )
        except:
            print(f'Fitting failed for shift {shift}')
            continue
        
        y_fit = grey_spectrum(df.iloc[:, 0], *popt)
        errors[i] = np.sum((sum_intensities - y_fit) ** 2)
        parameters[i] = popt

    # fitt with best parameters
    popt = parameters[np.argmin(errors)]
    y_fit = grey_spectrum(df.iloc[:, 0], *popt)
    print(popt)

    # print each lorentzian individually
    for i in range(len(positions)):
        plt.plot(df.iloc[:, 0], lorentzian(df.iloc[:, 0], popt[i], popt[i + len(positions)], popt[i + 2*len(positions)]))
    plt.show()




def main():
    file_names  = get_file_names()
    print(file_names)
    file_names = containing_string(file_names, 'Fumarate', 'ser')
    print(file_names)
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

        number_peaks = len(positions)
        y_fits = np.zeros((df.shape[0], df.shape[1]))
        fit_params = np.zeros((df.shape[1], number_peaks * 3))

        correct_starting_point(df, positions, names)
        '''for i in range(1, df.shape[1]):
            print(f'Fitting column {i/df.shape[1]*100:.2f}%')
            # perform fitting
            x = df.iloc[:,0]
            y = df.iloc[:,i]
            popt, pcov = curve_fit(grey_spectrum, x, y, p0 = [positions, [0.1]*len(positions), [1000]*len(positions)], maxfev=10000)
            y_fits[:,i-1] = grey_spectrum(x, *popt)

            fit_params[i-1] = popt

        output_direc = f'{file_name}_output'
        plot_single(x, y_fits, positions, names, file_name, df, output_direc)
        plot_time_dependece(y_fits, positions, names, fit_params, output_direc)
'''

def lorentzian(x, x0, gamma, A):
    '''
    x is the datapoint
    x0 is the peak position
    gamma is the width
    A is the amplitude
    '''
    return A * gamma / ((x - x0)**2 + gamma**2)

def grey_spectrum(x, *params):
    '''
    x is the independent variable array
    params should be a flattened list of x0, gamma, and A values
    '''
    n = len(params) // 3
    x0 = params[:n]
    gamma = params[n:2*n]
    A = params[2*n:3*n]
    
    y = np.zeros(len(x))
    for i in range(n):
        y += lorentzian(x, x0[i], gamma[i], A[i])
    return y

main()
# FA_20240215_2H_yeast_Pyruvate-d3_5.csv
# FA_20231113_2H Yeast_Pyruvate-d3_1.csv