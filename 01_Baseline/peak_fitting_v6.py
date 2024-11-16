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

            shift = fit_params[i, j]
            gamma = fit_params[i, len(positions) + j]
            A = fit_params[i, 2*len(positions) + j]

            # Plot the individual Lorentzian for this peak
            plt.plot(x, lorentzian(x, shift, gamma, A), linestyle='--', color=colors[j % len(colors)], label=f'Peak {names[j]}')
        # Add legend to avoid repetition
        plt.legend()
        
        # Save the plot
        plt.savefig(f'{output_direc}/{file_name}_{i}.png')
        plt.close()


def make_bounds(positions, names):
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
    return flattened_bounds, flattened_bounds_fine


def main():
    file_names  = get_file_names()
    #file_names = containing_string(file_names, 'Nicotinamide', 'ser') # debuging
    for file_name in file_names:
        # init dataframes and lists for results, this will contain  tuples of the form (df_fitting_params, df_spectrum)
        results = []
        print(f'Processing {file_name}')
        df = pd.read_csv(f'../Data/{file_name}')
        # meta path independet of the OS
        meta_path = Path('..', 'Data', 'Data_description_main.xlsx')
        # meta_path = '../Data/Data_description.xlsx'
        meta_df = pd.read_excel(meta_path)
        # extract ppm lines and names
        positions, names = extract_ppm_all(meta_df, file_name)

        if len(positions) == 0:
            print(f'No ppm values found for {file_name}')
            continue
        #####################################
        # Preparing shared height and width #
        #####################################

        flattened_bounds, flattened_bounds_fine = make_bounds(positions, names)

        number_peaks = len(positions)
        y_fits = np.zeros((df.shape[0], df.shape[1] - 1))
        # width and amplitude are shared for each metabolite peak, 
        # number of peaks for each metabolite extracting from the names file
        nr_ppm_pos = len(positions)
        nr_widths_pos = len(positions)
        nr_amplitudes_pos = len(positions)
        from copy import deepcopy
        names_modified = deepcopy(names)
        #               positions           widths              amplitudes
        mapping_names = deepcopy(names) + list(dict.fromkeys(names)) + list(dict.fromkeys(names))
        print('Mapping names: ', mapping_names) # need to have the same order as the parameters?
        for i in range(1, 6):
            '''
            - First n(peak number) positions
            - Next k width values
            - Next k amplitude values
            There are less width and amplitude values than positions, because they are shared between all peaks
            '''
            number_peaks = names.count('Metab{i}')
            nr_widths_pos -= number_peaks + 1 # 1 to have one shift left
            nr_amplitudes_pos -= number_peaks + 1
            # make a set of the names modifed list, but only apply the setting on the metabolites
        
        # also for substrates
        number_peaks = names.count('ReacSubs')
        nr_widths_pos -= number_peaks + 1
        nr_amplitudes_pos -= number_peaks + 1
        first_fit = True

        fit_params = np.zeros((df.shape[1] - 1, 3*len(positions)))

        ############################
        # Fitting each frame       #
        ############################
        print('Number of frames: ', df.shape[1])
        for i in range(1,df.shape[1]):
            try: # try in case some data can not be fitted
                print(f'Fitting column {i/df.shape[1]*100:.2f}%')
                x = df.iloc[:,0]
                y = df.iloc[:,i]
                # to increase fitting speed, increase tolerance
                if first_fit:
                    popt, pcov = curve_fit(lambda x, *params: grey_spectrum(x, positions, mapping_names ,*params),
                                        x, y, p0=[0] + [0.1]*len(list(dict.fromkeys(names))) + [1000]*len(list(dict.fromkeys(names))),
                                        maxfev=3000, ftol=1e-1, xtol=1e-1, bounds=flattened_bounds)
                    # init parameters for fine tuning
                    positions_fine = popt[0] + positions # das passt so, müsste immer noch funktionieren
                    widths = popt[1:len(list(set(names)))+1] 
                    amplitudes = popt[len(list(set(names)))+1:]
                    first_fit = False
                
                shift_lower_bounds_fine = positions_fine - 0.1

                # shift upper bounds fine-tune
                shift_upper_bounds_fine = positions_fine + 0.1

                n_positions_reduced = len(set(names))
                width_lower_bounds = np.full(n_positions_reduced, 0)
                # width upper bounds
                width_upper_bounds = np.full(n_positions_reduced, 1e-1)
                # amplitude lower bounds
                amplitude_lower_bounds = np.full(n_positions_reduced, 0)
                # amplitude upper bounds
                amplitude_upper_bounds = np.full(n_positions_reduced, np.inf)

                if first_fit:
                    lower_fine = np.concatenate([shift_lower_bounds_fine, width_lower_bounds, amplitude_lower_bounds])
                    upper_fine = np.concatenate([shift_upper_bounds_fine, width_upper_bounds, amplitude_upper_bounds])
                    flattened_bounds_fine = (lower_fine, upper_fine)

                # Fine tune the fit
                popt, pcov = curve_fit(lambda x, *params: grey_spectrum_fine_tune(x, np.array(positions), np.array(mapping_names), *params),
                                        x, y, p0= np.concatenate([positions_fine, widths, amplitudes]), maxfev=20000, bounds = flattened_bounds_fine, ftol=1e-5, xtol=1e-5)

                positions_fine = popt[:len(names)]
                widths = popt[len(names):2*len(names)]
                amplitudes = popt[2*len(names):]

                y_fits[:,i-1] = grey_spectrum_fine_tune(x,np.array(positions), np.array(mapping_names), *popt)
                # parameter verbreitern, damit wieder len(position) positionen, breiten und höhen rauskommen
                exp_pots = popt[:nr_ppm_pos]
                widths_final = []
                amplitudes_final = []
                k = 0
                dummy = names[k]
                for name in names:
                    if name != dummy:
                        k += 1
                        dummy = name
                    widths_final.append(popt[nr_ppm_pos + k])
                    amplitudes_final.append(popt[nr_ppm_pos + len(set(names)) + k])
                
                popt = np.concatenate([exp_pots, widths_final, amplitudes_final])
                fit_params[i-1] = popt

            except Exception:
                # what is the error
                import traceback
                traceback.print_exc()
                print('Fitting failed.')
                continue
        output_direc = f'output/6th_fit/{file_name}_output'
        #integrate_peaks(fit_params, names)
        plot_single(x, y_fits, positions, names, file_name, df, output_direc, fit_params)
        #plot_single_difference(x, y_fits, positions, names, file_name, df, output_direc, fit_params)
        plot_time_dependce(fit_params, file_name, output_direc, names)
        save_deep_learning_data(fit_params, names,positions,output_direc, file_name)

def save_deep_learning_data(fit_params, names, positions, output_direc, file_name):
    column_names = [f'{name}_pos_{pos}' for name, pos in zip(names, positions)] + [f'{name}_width_{pos}' for name, pos in zip(names, positions)] + [f'{name}_amp_{pos}' for name, pos in zip(names, positions)]
    df_results = pd.DataFrame(fit_params, columns=column_names)

    output_direc = '/'.join(output_direc.split('/')[:-1]) + '/fit_params/'
    os.makedirs(output_direc, exist_ok=True)
    df_results.to_json(f'{output_direc}{file_name}.json')


def plot_single_difference(x, y_fits, positions, names, file_name, df, output_direc, fit_params):
    # plot the difference between the fit and the data
    for i in range(y_fits.shape[1] - 1):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y_fits[:, i] - df.iloc[:, i + 1], label='Difference')
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Intensity')
        plt.title(f'Difference between fit and data of {file_name}')
        plt.legend()
        plt.savefig(f'{output_direc}/{file_name}_{i}_difference.png')
        plt.close()

def plot_time_dependce(fit_params, file_name, output_direc, names):
    # fit_paras is a 2d array, where the first dimension is the time and the second dimension is the fit parameters
    # fit_params[0] is the first time point
    # fit_params[0][0] is the first fit parameter of the first time point
    # ...
    #os.mkdir(output_direc)
    df_integrated = integrate_peaks(fit_params, names)

    plt.figure(figsize=(10, 6))
    # iterate over columns 
    for col_name in df_integrated.columns:
        plt.plot(df_integrated[col_name], label=col_name)
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.title(f'Intensity of peaks over time of {file_name}')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'{output_direc}/{file_name}_time_dependence.png')

def integrate_peaks(fit_params, names):
    integrated_values = np.zeros((fit_params.shape[0], len(set(names))))
    # sum names are multiple times in the names list, therefore the intenstys need to be summed up
    # mask the names
    print(names)
    print(fit_params[0])
    print(fit_params[0][2*len(names):])
    already_integrated = []
    for name in list(set(names)):
        mask = np.array(names) == name
        print(mask)
        if name not in already_integrated:
            for i in range(fit_params.shape[0]):
                integrated_values[i, list(set(names)).index(name)] = np.sum(fit_params[i][2*len(names):][mask])
        already_integrated.append(name)

    print(integrated_values)
    df_integrated = pd.DataFrame(integrated_values, columns=list(set(names)))
    print(df_integrated)
    return df_integrated
    
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


main()
