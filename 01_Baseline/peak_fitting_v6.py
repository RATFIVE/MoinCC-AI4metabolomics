'''
This model is an improvement of model 5. 
Only the first the fittings are done. On the consecutive fittings the previous fitting parameters are used as initial guess.
'''

import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
# fitting library
from scipy.optimize import curve_fit
import sys


# get file names
def get_file_names():
    """
    Get all filenames in the data directory. Using the Path library to make the code OS independent. Files need to end with .csv

    Returns:
        path_list: list of all filenames in the data directory
    """
    path_list = []
    # data_direc independent of the OS
    data_direc = Path('..','Data')
    # get all filenames which end with .csv
    for file in os.listdir(data_direc):
        if file.endswith('.csv'):
            path_list.append(file)
    return path_list

def containing_string(file_names, string = '', not_string = None):
    """
    Get all filenames which contain a specific string. If not_string is given, the string must be present and the not_string must not be present.

    Args:
        file_names: list of all filenames
        string: string which should be present in the filename
        not_string: string which should not be present in the filename

    Returns:
        list: list of all filenames which contain the string
    """
    # get all filenames which contain the string
    return [file for file in file_names if string in file and (not_string is None or not_string not in file)]

def extract_ppm_all(meta_df, file_name):
    """
    Extract the ppm values from the metadata file for a specific file.

    Args:
        meta_df: metadata dataframe
        file_name: name of the file
    
    Returns:
        positions: list of all ppm values
        names: list of all names of the ppm values
    """
    meta_df = meta_df[meta_df['File'].astype(str).str.upper() == str(file_name).upper()]
    if meta_df.shape[0] == 0:
        print(f'No metadata found for {file_name}')
        return [], []
    positions = []
    names = []

    react_substrat = str(meta_df['Substrate_ppm'].iloc[0]).split(',')
    for i in range(len(react_substrat)):
        names.append('ReacSubs')
        positions.append(float(react_substrat[i]))
  
    for i in range(1, 6):
        react_metabolite = str(meta_df[f'Metabolite_{i}_ppm'].iloc[0]).split(',')
        if react_metabolite == ['nan']:
            continue
        for j in range(len(react_metabolite)):
            print(react_metabolite)
            names.append(f'Metab{i}')
            positions.append(float(react_metabolite[j]))

    # water ppm
    positions.append(float(meta_df['Water_ppm'].iloc[0]))
    names.append('Water')

    return positions, names


def plot_single(x, y_fits, positions, names, file_name, df, output_direc, fit_params, fit_params_error, time_points):
    '''
    Plot the fitted data and the original data for each frame. The peaks are also plotted.

    Args:
        x: x values
        y_fits: fitted data
        positions: positions of the peaks, position and names have the same order and length. positions are corresponding to each other.
        names: names of the peaks, e.g. Metab1, Metab2, ...
        file_name: name of the file
        df: dataframe of the file
        output_direc: output directory
        fit_params: fitted parameters

    Returns:
        None
    '''
    # create output directory if it does not exist
    if not os.path.exists(output_direc):
        os.makedirs(output_direc)

    # plot the fitted data and the original data for each frame
    for i in range(y_fits.shape[1] - 1):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y_fits[:, i], label='Fit') # plot the fitted data
        plt.plot(df.iloc[:, 0], df.iloc[:, i + 1], label='Data')
        
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Intensity')
        plt.title(f'NMR Spectrum of {file_name} at {time_points[i]}s')
        
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

        # Add vertical lines for the ppm and plot the individual Lorentzians
        for j in range(len(positions)):
            # Add vertical lines for ppm positions
            #plt.axvline(x=positions[j], linestyle='--', color=colors[j % len(colors)], label=names[j])
            # Extract fitted parameters for this peak

            shift = fit_params[i, j]
            gamma = fit_params[i, len(positions) + j]
            A = fit_params[i, 2*len(positions) + j]




            # write fit params also in the plot
            pos = fit_params[i, j]
            width = fit_params[i, len(positions) + j]
            amplitude = fit_params[i, 2*len(positions) + j]
            pos_error = fit_params_error[i, j]
            width_error = fit_params_error[i, len(positions) + j]
            amplitude_error = fit_params_error[i, 2*len(positions) + j]

            # Add label to the plot, not used, because of too many labels
            label = (f"{names[j]}:"
                    f"$\\mathrm{{Center}} = {pos:.2f} \\pm {pos_error:.2f}$\n"
                    f"$\\gamma = {width:.2f} \\pm {width_error:.2f}$\n"
                    f"$A = {amplitude:.2f} \\pm {amplitude_error:.2f}$")

            # Plot the individual Lorentzian for this peak
            plt.plot(x, lorentzian(x, shift, gamma, A), linestyle='--', color=colors[j % len(colors)], label=f'{names[j]}')
        # Add legend to avoid repetition
        plt.legend()
        plt.grid()
        # Save the plot
        plt.savefig(f'{output_direc}/{file_name}_{time_points[i]}s.png')
        # get x and y limits
        x_lim = plt.xlim()
        y_lim = plt.ylim()
        plt.close()

        # also plot the difference between the fit and the data, use
        plt.figure(figsize=(10, 6))
        plt.plot(x, y_fits[:, i] - df.iloc[:, i + 1], label='Difference')
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Intensity')
        plt.title(f'Difference between fit and data of {file_name} at {time_points[i]}s')
        plt.grid()
        plt.legend()
        # set x and y limits
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.savefig(f'{output_direc}/{file_name}_{time_points[i]}s_difference.png')
        plt.close()

        # plot only fit
        plt.figure(figsize=(10, 6))
        plt.plot(x, y_fits[:, i], label='Fit') # plot the fitted data
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Intensity')
        plt.title(f'NMR Spectrum of {file_name} at {time_points[i]}s')
        plt.grid()
        plt.legend()
        plt.savefig(f'{output_direc}/{file_name}_{time_points[i]}s_fit.png')
        plt.close()
    


def make_bounds(positions, names, mode, positions_fine = None):
    if mode == 'first':
        # Assume positions is already defined
        n_positions_reduced = len(set(names))
        # width lower bounds
        width_lower_bounds = np.full(n_positions_reduced, 0)
        # width upper bounds
        width_upper_bounds = np.full(n_positions_reduced, 1e-1)
        # amplitude lower bounds
        amplitude_lower_bounds = np.full(n_positions_reduced, 0)
        # amplitude upper bounds
        amplitude_upper_bounds = np.full(n_positions_reduced, np.inf)
        # shift lower bounds, bounds not necessary since the shift is shared
        shift_lower_bounds = np.full(1, -np.inf)  # Single value, hence length 1
        # shift upper bounds
        shift_upper_bounds = np.full(1, np.inf)
        # combine bounds
        lower = np.concatenate([shift_lower_bounds, width_lower_bounds, amplitude_lower_bounds])
        upper = np.concatenate([shift_upper_bounds, width_upper_bounds, amplitude_upper_bounds])
        flattened_bounds = (lower, upper)
        shift_lower_bounds_fine = np.array(positions) - 0.1
        # shift upper bounds fine-tune
        shift_upper_bounds_fine = np.array(positions) + 0.1
        lower_fine = np.concatenate([shift_lower_bounds_fine, width_lower_bounds, amplitude_lower_bounds])
        upper_fine = np.concatenate([shift_upper_bounds_fine, width_upper_bounds, amplitude_upper_bounds])
        flattened_bounds_fine = (lower_fine, upper_fine)
        return flattened_bounds
    
    elif mode == 'fine':
        shift_lower_bounds_fine = positions_fine - 0.15
        # shift upper bounds fine-tun
        shift_upper_bounds_fine = positions_fine + 0.15
        n_positions_reduced = len(set(names))
        width_lower_bounds = np.full(n_positions_reduced, 0)
        # width upper bounds
        width_upper_bounds = np.full(n_positions_reduced, 2e-1)
        # amplitude lower bounds
        amplitude_lower_bounds = np.full(n_positions_reduced, 0)
        # amplitude upper bounds
        amplitude_upper_bounds = np.full(n_positions_reduced, np.inf)
        lower_fine = np.concatenate([shift_lower_bounds_fine, width_lower_bounds, amplitude_lower_bounds])
        upper_fine = np.concatenate([shift_upper_bounds_fine, width_upper_bounds, amplitude_upper_bounds])
        flattened_bounds_fine = (lower_fine, upper_fine)
        return flattened_bounds_fine

def get_meta_data(file_path):
    """
    Get the metadata from the file.

    Args:
        file_path: path to the metadata file

    Returns:
        meta_df: metadata dataframe
    """
    meta_df = pd.read_excel(file_path)
    return meta_df

def get_number_unique_peaks(names):
    """
    Returns the number of unique peaks in the names list. Some substances form multiple peaks.

    Args:
        names: list of all names

    Returns:
        int: number of unique peaks
    """
    return len(set(names))

def make_mapping_names(names):
    """
    Create a mapping of the names. The names are repeated multiple times, the mapping names are unique. Because multiple peaks of the same substance share the same height and width.

    Args:
        names: list of all names

    Returns:
        list: list of all unique mapping names
    """
    from copy import deepcopy
    return deepcopy(names) + list(dict.fromkeys(names)) + list(dict.fromkeys(names))

def unpack_params_errors(n_unique_peaks, number_peaks, names, popt, pcov):
    """
    Unpack the parameters and errors from the fitting. This is because the fitting was done with shared parameters. To get the individual parameters, the shared parameters need to be unpacked.

    Args:
        n_unique_peaks: number of unique peaks
        number_peaks: number of peaks
        names: list of all names
        popt: fitted parameters
        pcov: covariance matrix
    
    Returns:
        tuple: tuple of the unpacked parameters and errors
    """
    error = np.sqrt(np.diag(pcov))
    # needs  n_unique_peaks, number_peaks, names, popt
    widths_final = []
    amplitudes_final = []
    
    widths_final_error = []
    amplitudes_final_error = []
    k = 0
    dummy = names[k]
    for name in names:
        if name != dummy:
            k += 1
            dummy = name
        widths_final_error.append(error[number_peaks + k])
        amplitudes_final_error.append(error[number_peaks + n_unique_peaks + k])
        widths_final.append(popt[number_peaks + k])
        amplitudes_final.append(popt[number_peaks + n_unique_peaks + k])
                
    return np.concatenate([popt[:number_peaks], widths_final, amplitudes_final]), np.concatenate([error[:number_peaks], widths_final_error, amplitudes_final_error])

def main():
    
    file_names  = get_file_names()
    file_names = containing_string(file_names, 'Nicotinamide') # debuging
    for file_name in file_names:
        print(f'Processing {file_name}')

        # read and parse the metadata
        df = pd.read_csv(f'../Data/{file_name}')
        meta_df = get_meta_data(Path('..', 'Data', 'Data_description_main.xlsx'))
        positions, names = extract_ppm_all(meta_df, file_name)

        time_points = np.arange(0, df.shape[1] - 1) * meta_df[meta_df['File'] == file_name]['TRtotal[s]'].iloc[0]
        fit_params = np.zeros((df.shape[1] - 1, 3*len(positions)))
        fit_params_error = np.zeros((df.shape[1] - 1, 3*len(positions)))
        y_fits = np.zeros((df.shape[0], df.shape[1] - 1))

        output_direc = f'output/6th_fit/{file_name}_output'

        if len(positions) == 0:
            print(f'No ppm values found for {file_name}')
            continue
        
        # bounds for the first fitting, which corresponds to the first frame
        flattened_bounds = make_bounds(positions, names, mode='first')

        number_peaks = len(positions)
        mapping_names = make_mapping_names(names)
        n_unique_peaks = get_number_unique_peaks(names)

        first_fit = True
        # x values are the first column, same for all frames
        x = df.iloc[:,0]
        # iterate over all time points
        for i in range(1,df.shape[1]):
            try: # try in case some data can not be fitted
                print(f'Fitting column {i/df.shape[1]*100:.2f}%')
                y = df.iloc[:,i]
                # to increase fitting speed, increase tolerance
                if first_fit:
                    popt, pcov = curve_fit(lambda x, *params: grey_spectrum(x, positions, mapping_names ,*params),
                                        x, y, p0=[0] + [0.1]*len(list(dict.fromkeys(names))) + [1000]*len(list(dict.fromkeys(names))),
                                        maxfev=3000, ftol=1e-1, xtol=1e-1, bounds=flattened_bounds)
                    
                    # init parameters for fine tuning
                    positions_fine = popt[0] + positions
                    widths = popt[1:n_unique_peaks+1] 
                    amplitudes = popt[n_unique_peaks+1:]

                    # starting parameters for fine tuning
                    p0 = np.concatenate([positions_fine, widths, amplitudes])
                    # bounds for fine tuning
                    flattened_bounds_fine = make_bounds(positions, names, 'fine', positions_fine)

                    first_fit = False
            
                # Fine tune the fit
                popt, pcov = curve_fit(lambda x, *params: grey_spectrum_fine_tune(x, np.array(positions), np.array(mapping_names), *params),
                                        x, y, p0= np.concatenate([positions_fine, widths, amplitudes]), maxfev=20000, bounds = flattened_bounds_fine, ftol=1e-3, xtol=1e-3)

                positions_fine = popt[:number_peaks]
                widths = popt[number_peaks:number_peaks + n_unique_peaks]
                amplitudes = popt[number_peaks + n_unique_peaks:]
                

                y_fits[:,i-1] = grey_spectrum_fine_tune(x,np.array(positions_fine), np.array(mapping_names), *popt)

                # unpack the parameters and errors
                fit_params[i-1], fit_params_error[i-1] = unpack_params_errors(n_unique_peaks, number_peaks, names, popt, pcov)

            except RuntimeError:
                print(f'Could not fit time frame number {i}. Skipping...')

        plot_single(x, y_fits, positions, names, file_name, df, output_direc, fit_params, fit_params_error, time_points)
        plot_time_dependence(fit_params, file_name, output_direc, names, fit_params_error, time_points)

def plot_time_dependence(fit_params, file_name, output_direc, names, fit_params_error, time_points):
    """
    Plot the intensity of the peaks over time.

    Args:
        fit_params: fitted parameters
        file_name: name of the file
        output_direc: output directory
        names: list of all names
        fit_params_error: errors of the fitted parameters
        time_points: time points
    """
    df_integrated, df_integrated_error = integrate_peaks(fit_params, names, fit_params_error)
    plt.figure(figsize=(10, 6))
    # iterate over columns 
    for col_name in df_integrated.columns:
        plt.errorbar(time_points,df_integrated[col_name], label=col_name, yerr=df_integrated_error[col_name])
    plt.xlabel('Time in s')
    plt.ylabel('Intensity')
    plt.grid()
    plt.title(f'Intensity of peaks over time of {file_name}')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'{output_direc}/{file_name}_time_dependence.png')

def integrate_peaks(fit_params, names, fit_params_error):
    """
    Integrate the peaks over time. The integral of the peak is the peak height. Multiple peaks of the same substance are summed up.

    Args:
        fit_params: fitted parameters
        names: list of all names
        fit_params_error: errors of the fitted parameters
    
    Returns:
        tuple: tuple of the integrated values and the errors
    """
    integrated_values = np.zeros((fit_params.shape[0], len(set(names))))
    integrated_values_error = np.zeros((fit_params.shape[0], len(set(names))))
    # sum names are multiple times in the names list, therefore the intensitys need to be summed up
    # mask the names
    already_integrated = []
    for name in list(set(names)):
        mask = np.array(names) == name
        print(mask)
        if name not in already_integrated:
            for i in range(fit_params.shape[0]):
                integrated_values[i, list(set(names)).index(name)] = np.sum(fit_params[i][2*len(names):][mask])
                integrated_values_error[i, list(set(names)).index(name)] = np.sum(fit_params_error[i][2*len(names):][mask])
        already_integrated.append(name)

    df_integrated = pd.DataFrame(integrated_values, columns=list(set(names)))
    df_integrated_error = pd.DataFrame(integrated_values_error, columns=list(set(names)))
    return df_integrated, df_integrated_error
    
def lorentzian(x, shift, gamma, A):
    '''
    x is the datapoint
    x0 is the peak position
    gamma is the width
    A is the amplitude
    '''
    return A * gamma / ((x - shift)**2 + gamma**2)


def grey_spectrum(x, positions,mapping_names,*params):
    '''
    x is the independent variable array
    params should be a flattened list of shift, gamma, and A values
    '''
    shift = params[0]            # Single shift parameter
    number_unique_substrates = len(set(mapping_names)) 
    gamma = params[1:number_unique_substrates+1]        # Extract n gamma values
    A = params[number_unique_substrates+1:]             # Extract n A values
    n = len(positions)

    y = np.zeros(len(x))
    k = 0
    current_name = mapping_names[0] # not 0, it must be the first name from second ot third part of the mapping_names
    for i in range(n):
        # retrieve gamma and A values
        # Peak position is shared between all peaks
        if mapping_names[i] != current_name:
            k += 1
            current_name = mapping_names[i]
        if k < number_unique_substrates:
            y += lorentzian(x, shift + positions[i], gamma[k], A[k])

    # stop code from execution
    return y

def grey_spectrum_fine_tune(x,positions,mapping_names,*params):
    '''
    x is the independent variable array
    params should be a flattened list of x0, gamma, and A values
    '''
    n = len(positions)
    n_reduced = len(set(mapping_names))
    x0 = params[:n]
    gamma = params[n:n + n_reduced]
    A = params[n+n_reduced:]

    y = np.zeros(len(x))
    k = 0
    current_name = mapping_names[0]
    for i in range(n):
        if mapping_names[i] != current_name:
            k += 1
            current_name = mapping_names[i]
        if k < n_reduced:
            y += lorentzian(x, x0[i], gamma[k], A[k])
    return y


# For deep learning model. save params to json
#def save_deep_learning_data(fit_params, names, positions, output_direc, file_name):
#    column_names = [f'{name}_pos_{pos}' for name, pos in zip(names, positions)] + [f'{name}_width_{pos}' for name, pos in zip(names, positions)] + [f'{name}_amp_{pos}' for name, pos in zip(names, positions)]
#    df_results = pd.DataFrame(fit_params, columns=column_names)
#
#    output_direc = '/'.join(output_direc.split('/')[:-1]) + '/fit_params/'
#    os.makedirs(output_direc, exist_ok=True)
#    df_results.to_json(f'{output_direc}{file_name}.json')

main()
