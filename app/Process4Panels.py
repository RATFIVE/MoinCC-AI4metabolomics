import os
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class Process4Panels:
    def __init__(self, data_file_path):
        # prepare file paths
        self.data_file_path = data_file_path
        self.file_name = os.path.basename(data_file_path)
        self.output_dir = Path('output', self.file_name + '_output')
        self.fitting_params_fp = Path(str(self.output_dir), 'fitting_params.csv')
        self.fitting_params_err_fp = Path(str(self.output_dir), 'fitting_params_error.csv')

        # read in necessary data
        self.data = pd.read_csv(self.data_file_path)
        self.fitting_params = pd.read_csv(self.fitting_params_fp)
        self.fitting_params_error = pd.read_csv(self.fitting_params_err_fp)

    def lorentzian(self, x, shift, gamma, A, y_shift):
        '''
        x is the datapoint
        x0 is the peak position
        gamma is the width
        A is the amplitude
        '''
        return A * gamma / ((x - shift)**2 + gamma**2) + y_shift
    
    def save_sum_spectra(self):
        x = self.data.iloc[:,0]
        self.sum_df = pd.DataFrame({'x': x})
        for i, row in enumerate(self.fitting_params.iterrows()):
            n_peaks = int((len(row[1]) - 2) / 3)
            y_shift = row[1].iloc[1]
            positions = row[1][2:n_peaks+2]
            widths = row[1][2+n_peaks: 2*n_peaks+2]
            amplitudes = row[1][2*n_peaks+2:]
            y = np.zeros(len(x))
            for position,width, amplitude in zip(positions, widths, amplitudes):
                y += self.lorentzian(x, position, width, amplitude, y_shift)
            y = pd.DataFrame({str(i): y})
            self.sum_df = pd.concat([self.sum_df, y] ,axis=1)
        self.sum_df.set_index('x')
        self.sum_df.to_csv(str(Path(str(self.output_dir), 'sum_fit.csv')),index=False)
    
    def save_substrate_individual(self):
        x = self.data.iloc[:,0]
        output_path = Path(str(self.output_dir), 'substance_fits')
        os.makedirs(str(output_path), exist_ok=True)

        for i, row in enumerate(self.fitting_params.iterrows()):
            n_peaks = int((len(row[1]) - 2) / 3)
            names = row[1].index[2:]  # This remains the same because `index` is used for column labels
            y_shift = row[1].iloc[1]
            positions = row[1].iloc[2:n_peaks + 2]
            widths = row[1].iloc[2 + n_peaks: 2 * n_peaks + 2]
            amplitudes = row[1].iloc[2 * n_peaks + 2:]
            time_frame_res = pd.DataFrame({'x': x})
            
            # group positons according to substances
            substances = set([name.split('_')[0] for name in names])
            for substance in substances:
                indices = []
                # get positions 
                indices = [idx for idx, name in enumerate(names[:n_peaks]) if substance in name]

                rel_pos = positions.iloc[indices]
                rel_width = widths.iloc[indices]
                rel_ampl = amplitudes.iloc[indices]

                y = np.zeros(len(x))
                for position,width, amplitude in zip(rel_pos, rel_width, rel_ampl):
                    y += self.lorentzian(x, position,width, amplitude, y_shift)
                y = pd.DataFrame({substance : y})
                time_frame_res = pd.concat([time_frame_res, y], axis = 1)
            
            time_frame_res.to_csv(str(Path(str(output_path), f'sum_fit{i}.csv')), index=False)

    def save_difference(self):
        # Extract the x-axis data
        x = self.data.iloc[:, 0]
        # Initialize differences DataFrame with x as the first column
        differences = pd.DataFrame({'x': x})
        # Compute differences for each column and concatenate
        diff_columns = [self.data.iloc[:, i] - self.sum_df.iloc[:, i] for i in range(1, self.sum_df.shape[1])]

        diff_df = pd.concat(diff_columns, axis=1)
        diff_df.columns = [str(i) for i in range(0, self.sum_df.shape[1]-1)]
        differences = pd.concat([differences, diff_df], axis=1)
        differences.to_csv(Path(self.output_dir) / 'differences.csv', index=False)
    
    def save_kinetics(self):
        n_peaks = self.fitting_params.shape[1] // 3
        names = self.fitting_params.columns[2:] 

        substances = list(set([name.split('_')[0] for name in names]))

        kinetics = np.zeros((self.fitting_params.shape[0],len(substances)))

        # the integral is just the amplitude
        for i, row in enumerate(self.fitting_params.iterrows()):
            amplitudes = row[1].iloc[2 * n_peaks + 2:]
            # group positons according to substances, unnecessary loop. complete waste of computational resources
            for k, substance in enumerate(substances):
                indices = []
                # get positions 
                indices = [idx for idx, name in enumerate(names[:n_peaks]) if substance in name]
                # row i and columns k
                kinetics[i,k] = sum(amplitudes.iloc[indices].values)
        kinetics = pd.DataFrame(kinetics, columns = substances)
        kinetics['time step'] = np.arange(0, kinetics.shape[0])
        kinetics.to_csv(Path(self.output_dir, 'kinetics.csv'),index=False)

# processer = Process4Panels('/home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/Data/FA_20240213_2H_yeast_Fumarate-d2_9.csv')
# processer.save_sum_spectra()
# processer.save_substrate_individual()
# processer.save_difference()
# processer.save_kinetics()

# output/FA_20240731_2H_yeast_Fumarate-d2_15_200.ser.csv_output/fitting_params.csv
# output/FA_20240731_2H_yeast_Fumarate-d2_15_200.ser.csv_output/fitting_params.csv