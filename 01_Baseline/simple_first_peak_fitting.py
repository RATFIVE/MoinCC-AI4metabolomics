import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
# fitting library
from scipy.optimize import curve_fit

'''
NOT WORKING YET
'''

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

def containing_string(file_names, string):
    # get all filenames which contain the string
    return [file for file in file_names if string in file]

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
    react_metabolite = str(meta_df['Metabolite_1_ppm'].iloc[0]).split(',')
    for i in range(len(react_metabolite)):
        names.append('Metab1')
        positions.append(float(react_metabolite[i]))

    # water ppm
    positions.append(float(meta_df['Water_chemical shift (ppm)'].iloc[0]))
    names.append('Water')
    return positions, names

def single_plot(df, y, ppm_lines, names, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(df[:, 0], df[:, y])
    plt.xlabel('Chemical Shift (ppm)')
    plt.ylabel('Intensity')
    plt.title(f'NMR Spectrum of {file_name}')
    # add vertical lines for the ppm
    for i in range(len(ppm_lines)):
        plt.axvline(x=ppm_lines[i], linestyle='--', label=names[i])
    plt.legend()
    plt.show()


def main():
    #file_names = get_file_names()
    #file_names = containing_string(file_names, 'Fumerate')
    file_name = 'FA_20240213_2H_yeast_Fumarate-d2_9.csv'
    print(f'Processing {file_name}')
    df = pd.read_csv(f'../Data/{file_name}')
    df = df.to_numpy()
    ppm_lines, names = extract_ppm_all(meta_df, file_name)
    single_plot(df, 40, ppm_lines, names, file_name)

def lorentzian(x, x0, gamma, A):
    '''
    x is the datapoint
    x0 is the peak position
    gamma is the width
    A is the amplitude
    '''
    return A * gamma**2 / ((x - x0)**2 + gamma**2)

def grey_spectrum(x, x0a, gammaa, Aa, x0b, gammab, Ab, x0c, gammac, Ac, x0d, gammad, Ad):
    return lorentzian(x, x0a, gammaa, Aa) + lorentzian(x, x0b, gammab, Ab) + lorentzian(x, x0c, gammac, Ac) + lorentzian(x, x0d, gammad, Ad)

def fit_lorentzian(x, y, x0s):
    # 4 lorentz function are fitted now
    # initial x0 value sfrom metadata
    popt, pcov = curve_fit(grey_spectrum, x, y, p0=x0s)

# get current working directory
cwd = os.getcwd()
# meta path independet of the OS
meta_path = Path('..', 'Data', 'Data_description.xlsx')
#meta_path = '../Data/Data_description.xlsx'
meta_df = pd.read_excel(meta_path)

main()