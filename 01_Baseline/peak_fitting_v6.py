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

            shift = fit_params[i, j]
            gamma = fit_params[i, len(positions) + j]
            A = fit_params[i, 2*len(positions) + j]

            # print('---------------------------------')
            # print('Shift:', shift)
            # print('Gamma:', gamma)
            # print('A:', A)
            # print('---------------------------------')

            # Plot the individual Lorentzian for this peak
            plt.plot(x, lorentzian(x, shift, gamma, A), linestyle='--', color=colors[j % len(colors)], label=f'Peak {names[j]}')

        #no also plot each individual
        # Add legend to avoid repetition
        plt.legend()

        # Save the plot
        plt.savefig(f'{output_direc}/{file_name}_{i}.pdf')
        plt.close()



def main():
    file_names  = get_file_names()
    #file_names = containing_string(file_names, 'Nicotinamide', 'ser')
    for file_name in file_names:
        print(f'Processing {file_name}')
        df = pd.read_csv(f'../Data/{ file_name}')
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

        # Assume positions is already defined
        n_positions = len(positions)
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

        # shift lower bounds fine-tune
        print('Positions:', positions)
        shift_lower_bounds_fine = np.array(positions) - 0.1

        # shift upper bounds fine-tune
        shift_upper_bounds_fine = np.array(positions) + 0.1


        lower_fine = np.concatenate([shift_lower_bounds_fine, width_lower_bounds, amplitude_lower_bounds])
        upper_fine = np.concatenate([shift_upper_bounds_fine, width_upper_bounds, amplitude_upper_bounds])

        flattened_bounds_fine = (lower_fine, upper_fine)

        number_peaks = len(positions)
        y_fits = np.zeros((df.shape[0], df.shape[1]))
        # width and amplitude are shared for each metabolite peak, 
        # number of peaks for each metabolite extracting from the names file
        nr_ppm_pos = len(positions)
        nr_widths_pos = len(positions)
        nr_amplitudes_pos = len(positions)
        from copy import deepcopy
        names_modified = deepcopy(names)
        #               positions           widths              amplitudes
        mapping_names = deepcopy(names) + list(dict.fromkeys(names)) + list(dict.fromkeys(names))
        # print('---------------------------------')  
        # print('Names:', names)
        # print('Mapping names:', mapping_names)
        # print('---------------------------------')
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
        fit_params = np.zeros((df.shape[1], 3*len(positions)))
        #fit_params = np.zeros((df.shape[1], number_peaks * 3))
        for i in range(1,df.shape[1]):
            try:
                print(f'Fitting column {i/df.shape[1]*100:.2f}%')
                # perform fitting
                x = df.iloc[:,0]
                y = df.iloc[:,i]
                # to increase fitting speed, increase tolerance
                if first_fit:
                    popt, pcov = curve_fit(lambda x, *params: grey_spectrum(x, positions, mapping_names, *params),
                                    x, y, p0=[0] + [0.1]*len(list(dict.fromkeys(names))) + [1000]*len(list(dict.fromkeys(names))),
                                    maxfev=3000, ftol=1e-1, xtol=1e-1, bounds=flattened_bounds)
                    first_fit = False
                    # init parameters for fine tuning
                    positions_fine = popt[0] + positions # das passt so, müsste immer noch funktionieren
                    widths = popt[1:len(list(set(names)))+1] # das muss angepasst werden
                    amplitudes = popt[len(list(set(names)))+1:]
                else:
                    


                shift_lower_bounds_fine = positions_fine - 0.1

                # shift upper bounds fine-tune
                shift_upper_bounds_fine = positions_fine + 0.1


                lower_fine = np.concatenate([shift_lower_bounds_fine, width_lower_bounds, amplitude_lower_bounds])
                upper_fine = np.concatenate([shift_upper_bounds_fine, width_upper_bounds, amplitude_upper_bounds])

                flattened_bounds_fine = (lower_fine, upper_fine)
                # Fine tune the fit
                popt, pcov = curve_fit(lambda x, *params: grey_spectrum_fine_tune(x, np.array(positions), np.array(mapping_names), *params),
                                        x, y, p0= np.concatenate([positions_fine, widths, amplitudes]), maxfev=20000, bounds= flattened_bounds_fine, ftol=1e-3, xtol=1e-3)
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
                    # print('---------------------------------')
                    # print('name', name)
                    # print('dummy', dummy)
                    # print('k', k)
                    # print('Popt:', popt)
                    # print('length positions', nr_ppm_pos)

                    # print('length widths', nr_ppm_pos + len(set(names)) + k)
                    # print('length amplitudes', nr_ppm_pos + len(set(names)) + k)
                    # print('---------------------------------')
                    widths_final.append(popt[nr_ppm_pos + k])
                    amplitudes_final.append(popt[nr_ppm_pos + len(set(names)) + k])
                popt = np.concatenate([exp_pots, widths_final, amplitudes_final])
#                print('Popt:', popt)
                fit_params[i-1] = popt
                    
                # print('---------------------------------')
                # print('fit_params:', fit_params[i-1])
                # print('Popt: ',popt)
                # fit_params[i-1] = popt
                # print('---------------------------------')
        

            except Exception:
                # what is the error
                import traceback
                traceback.print_exc()
                print('Fitting failed.')
                continue
        output_direc = f'output/5th_fit/{file_name}_output'
        plot_single(x, y_fits, positions, names, file_name, df, output_direc, fit_params)
        # stop


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

    # print('Mapping names:', mapping_names)
    # print('Params:', params)
    # print('Shift:', shift)
    # print('Number unique substrates:', number_unique_substrates)
    # print('Gamma:', gamma)
    # print('A:', A)

    y = np.zeros(len(x))
    k = 0
    current_name = mapping_names[0] # not 0, it must be the first name from second ot third part of the mapping_names
    for i in range(n):
        if mapping_names[i] != current_name:
            k += 1
            current_name = mapping_names[number_unique_substrates + i]
        # retrieve gamma and A values

        # Peak position is shared between all peaks
        y += lorentzian(x, shift + positions[i], gamma[k], A[k])
    return y

def grey_spectrum_fine_tune(x,positions,mapping_names,*params):
    '''
    x is the independent variable array
    params should be a flattened list of x0, gamma, and A values
    '''
    # print('---------------------------------')
    # print('Params:', params)
    # print('Positions:', positions)
    # print('Mapping names:', mapping_names)
    # print('---------------------------------')
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
            current_name = mapping_names[n_reduced + i]
        y += lorentzian(x, x0[i], gamma[k], A[k])
    return y


main()
