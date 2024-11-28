'''
This model is an improvement of model 5. 
Only the first the fittings are done. On the consecutive fittings the previous fitting parameters are used as initial guess.
'''


import pandas as pd
import os
import numpy as np
# fitting library
from scipy.optimize import curve_fit
from copy import deepcopy
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# import streamlit loadbar
import streamlit as st

class PeakFitting:
    def __init__(self, fp_file, fp_meta):
        # file paths
        self.fp_file = fp_file
        self.fp_meta = fp_meta
        # name of the data file and metadata file
        self.file_name = os.path.basename(fp_file)
        self.meta_name = os.path.basename(fp_meta)

        # define and create output directory
        self.output_direc = 'output/' + self.file_name + '_output/'
        os.makedirs(self.output_direc, exist_ok=True)

        # read in the files
        self.df = pd.read_csv(fp_file)
        self.meta_df = pd.read_excel(fp_meta)

        self.number_time_points = self.df.shape[1] - 1
        self.time_points = np.arange(1, self.number_time_points + 1) 
        self.x = self.df.iloc[:,0]

        # positions and corresponding names of the peaks
        self.positions, self.names = self.extract_ppm_all()
        self.number_peaks = len(self.positions)
        self.number_substances = len(set(self.names))

         # initialize outputs
        column_names =  ['Time'] + ['y_shift'] + [f'{name}_pos_{pos}' for name, pos in zip(self.names, self.positions)] + [f'{name}_width_{pos}' for name, pos in zip(self.names, self.positions)] + [f'{name}_amp_{pos}' for name, pos in zip(self.names, self.positions)]

        self.fitting_params = pd.DataFrame(columns=column_names).set_index('Time')
        self.fitting_params_error = pd.DataFrame(columns=column_names).set_index('Time')

        self.names_substances =  deepcopy(self.names) + list(dict.fromkeys(self.names)) + list(dict.fromkeys(self.names)) # Not a relevant instance attribute, so putting somewhere else?
        
    def extract_ppm_all(self):
        """
        Extract the ppm values from the metadata file for a specific file.

        Args:
            meta_df: metadata dataframe
            file_name: name of the file
        
        Returns:
            positions: list of all ppm values
            names: list of all names of the ppm values
        """
        self.meta_df = self.meta_df[self.meta_df['File'].astype(str).str.upper() == str(self.file_name).upper()]

        if self.meta_df.shape[0] == 0: #no metabolites listed --> only water present
            print(f'No metadata found for {self.file_name}')
        

        positions = []
        names = []
        react_substrat = str(self.meta_df['Substrate_ppm'].iloc[0]).split(',')
        if react_substrat != ['nan']:
            for i in range(len(react_substrat)):
                names.append('ReacSubs')
                positions.append(float(react_substrat[i]))

        for i in range(1, 6):
            react_metabolite = str(self.meta_df[f'Metabolite_{i}_ppm'].iloc[0]).split(',')
            if react_metabolite == ['nan']:
                continue
            for j in range(len(react_metabolite)):
                names.append(f'Metab{i}')
                positions.append(float(react_metabolite[j]))

        # water ppm
        positions.append(float(self.meta_df['Water_ppm'].iloc[0]))
        names.append('Water')

        return positions, names

    def make_bounds(self, mode, positions_fine = None,
                    y_shift = (0, np.inf),
                    shift_bounds = (-np.inf, np.inf),width_bounds = (0, 3e-1), amplitude_bounds = (0, np.inf),
                    shift_bounds_fine = (- 0.1, 0.1), width_bounds_fine = (0, 3e-1), amplitude_bounds_fine = (0, np.inf)):
        if mode == 'first':
            y_shift_lower_bounds = np.full(1, y_shift[0])
            y_shift_upper_bounds = np.full(1, y_shift[1])
            shift_lower_bounds = np.full(1, shift_bounds[0])  # shifting the whole spectrum
            shift_upper_bounds = np.full(1, shift_bounds[1])
            width_lower_bounds = np.full(self.number_substances, width_bounds[0])
            width_upper_bounds = np.full(self.number_substances, width_bounds[1])
            amplitude_lower_bounds = np.full(self.number_substances, amplitude_bounds[0])
            amplitude_upper_bounds = np.full(self.number_substances, amplitude_bounds[1])
            return (np.concatenate([y_shift_lower_bounds,shift_lower_bounds, width_lower_bounds, amplitude_lower_bounds]), np.concatenate([y_shift_upper_bounds,shift_upper_bounds, width_upper_bounds, amplitude_upper_bounds]))
        
        elif mode == 'fine':
            y_shift_lower_bounds = np.full(1, y_shift[0])
            y_shift_upper_bounds = np.full(1, y_shift[1])
            shift_lower_bounds_fine = positions_fine + shift_bounds_fine[0]
            # shift upper bounds fine-tun
            shift_upper_bounds_fine = positions_fine + shift_bounds_fine[1]
            width_lower_bounds = np.full(self.number_substances, width_bounds_fine[0])
            width_upper_bounds = np.full(self.number_substances, width_bounds_fine[1])
            # amplitude lower bounds
            amplitude_lower_bounds = np.full(self.number_substances, amplitude_bounds_fine[0])
            # amplitude upper bounds
            amplitude_upper_bounds = np.full(self.number_substances, amplitude_bounds_fine[1])

            return (np.concatenate([y_shift_lower_bounds, shift_lower_bounds_fine, width_lower_bounds, amplitude_lower_bounds]), np.concatenate([y_shift_upper_bounds, shift_upper_bounds_fine, width_upper_bounds, amplitude_upper_bounds]))


    def unpack_params_errors(self, popt, pcov):
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
        dummy = self.names[k]
        for name in self.names:
            if name != dummy:
                k += 1
                dummy = name
            widths_final_error.append(error[self.number_peaks + k + 1])
            amplitudes_final_error.append(error[self.number_peaks + self.number_substances + k + 1])
            widths_final.append(popt[self.number_peaks + k + 1])
            amplitudes_final.append(popt[self.number_peaks + self.number_substances + k + 1])
            
        return np.concatenate([np.array([popt[0]]), popt[1:self.number_peaks+1], widths_final, amplitudes_final]), \
            np.concatenate([np.array([error[0]]), error[1:self.number_peaks+1], widths_final_error, amplitudes_final_error])

    def fit(self, save_csv = True):
        # bounds for the first fitting, which corresponds to the first frame
        flattened_bounds = self.make_bounds(mode='first')
        
        first_fit = True
        # iterate over all time points
        for i in tqdm(range(self.number_time_points), desc= self.file_name):
            try: # try in case some data can not be fitted
                y = self.df.iloc[:,i+1]
                # to increase fitting speed, increase tolerance 
                if first_fit:
                    popt, pcov = curve_fit(lambda x, *params: self.grey_spectrum(x,*params),
                                        self.x, y, p0=[0] + [0] + [0.1]*self.number_substances + [1000]*self.number_substances,
                                        maxfev=3000, ftol=1e-1, xtol=1e-1, bounds=flattened_bounds)
                    
                    y_shift = np.array([popt[0]])
                    # init parameters for fine tuning
                    positions_fine = popt[1] + self.positions
                    widths = popt[2:self.number_substances+2] 
                    amplitudes = popt[self.number_substances+2:]

                    # starting parameters for fine tuning
                    p0 = np.concatenate([y_shift, positions_fine, widths, amplitudes])
                    # bounds for fine tuning
                    flattened_bounds_fine = self.make_bounds(mode = 'fine', positions_fine = positions_fine)
                    first_fit = False

                # Fine tune the fit
                popt, pcov = curve_fit(lambda x, *params: self.grey_spectrum_fine_tune(x, *params),
                                        self.x, y, p0 = p0, maxfev=20000, bounds = flattened_bounds_fine, ftol=1e-6, xtol=1e-6)

                y_shift = np.array([popt[0]])
                positions_fine = popt[1:self.number_peaks+1]
                widths = popt[1+self.number_peaks:self.number_peaks + self.number_substances+1]
                amplitudes = popt[self.number_peaks + self.number_substances+1:]

                p0 = np.concatenate([y_shift, positions_fine, widths, amplitudes])
                
                # unpack the parameters and errors
                self.fitting_params.loc[i], self.fitting_params_error.loc[i] = self.unpack_params_errors(popt, pcov)
                print(self.fitting_params.loc[i])
                # row i of dataframe is filled with the fitting parameters
                #print(self.fitting_params.loc[i])
                #sys.exit()
                
            except RuntimeError:
                print(f'Could not fit time frame number {i}. Skipping...')
        
        # save results
        if save_csv == True:
            self.fitting_params.to_csv(self.output_direc + 'fitting_params.csv')
            self.fitting_params_error.to_csv(self.output_direc + 'fitting_params_error.csv')
        else:
            return self.fitting_params

    
    def lorentzian(self, x, shift, gamma, A):
        '''
        x is the datapoint
        x0 is the peak position
        gamma is the width
        A is the amplitude
        '''
        return A * gamma / ((x - shift)**2 + gamma**2)

    # this has high potential for being wrong 
    def grey_spectrum(self, x, *params):
        '''
        x is the independent variable array
        params should be a flattened list of shift, gamma, and A values
        '''
        y_shift = params[0]           
        shift = params[1]            # Single shift parameter
        gamma = params[2:self.number_substances+2]        # Extract n gamma values
        A = params[self.number_substances+2:]             # Extract n A values

        y = np.zeros(len(x))
        k = 0
        current_name = self.names_substances[0]
        for i in range(self.number_peaks):
            # retrieve gamma and A values
            # Peak position is shared between all peaks
            if self.names[i] != current_name:
                k += 1
                current_name = self.names_substances[i]
            if k < self.number_peaks:
                y += self.lorentzian(x, shift + self.positions[i], gamma[k], A[k]) + y_shift
        # stop code from execution
        return y
    
    def write_results(self):
        self.fitting_params.to_csv(self.output_direc + 'fitting_params.csv')
        self.fitting_params_error.to_csv(self.output_direc + 'fitting_params_error.csv')

    # this has high potential for being wrong
    def grey_spectrum_fine_tune(self, x, *params):
        '''
        x is the independent variable array
        params should be a flattened list of x0, gamma, and A values
        '''
        y_shift = params[0]
        x0 = params[1:self.number_peaks+1]
        gamma = params[1+ self.number_peaks:self.number_peaks + self.number_substances+1]
        A = params[self.number_peaks+self.number_substances+1:]

        y = np.zeros(len(x))
        k = 0
        current_name = self.names_substances[0]
        for i in range(self.number_peaks):
            if self.names[i] != current_name:
                k += 1
                current_name = self.names_substances[i]
            if k < self.number_substances:
                y += self.lorentzian(x, x0[i], gamma[k], A[k]) + y_shift
        return y

# FA_20240207_2H_yeast_Fumarate-d2_5.csv
#input_file = '../Data/FA_20240213_2H_yeast_Fumarate-d2_9.csv'
#meta_file =  '/home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/Data/Data_description_main.xlsx'
###
#pf = PeakFitting(input_file, meta_file)
#pf.fit()
# Error Handling: Are filepathe existing?