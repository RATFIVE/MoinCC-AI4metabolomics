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


def plot_single(x, y_fits, positions, names, file_name, df):
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
        
        # Save the plot
        plt.savefig(f'output/{file_name}_{i}.png')
        plt.close()

def plot_time_dependece(y_fits, positions, names, popt):
    # the area of the indivudal peaks is the concentration with some kind of linear transformation on top.
    area_per_peak = np.zeros((y_fits.shape[1], len(positions)))
    print(area_per_peak.shape)
    for i in range(y_fits.shape[1]):
        for j in range(len(positions)):
            area_per_peak[i, j] = popt[i, 3*j + 2] * popt[i, 3*j+1]
    
    print(popt)
    print('Shape of area_per_peak:', area_per_peak.shape)
    print('Shape of y_fits:', y_fits.shape)

    time_points = range(y_fits.shape[1])

    # scatter plot of time evolution of the peak areas
    plt.figure(figsize=(10, 6))
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    for i in range(len(positions)):
        plt.scatter(time_points, area_per_peak[:, i], label=names[i], color=colors[i % len(colors)])
    plt.xlabel('Time')
    plt.ylabel('Area')
    plt.title('Time evolution of peak areas')
    plt.legend()
    plt.savefig(f'output/Time_evolution_areas.png')
    plt.close()    

def main():
    #file_names = containing_string(file_names, 'Fumerate')
    file_name = 'FA_20240213_2H_yeast_Fumarate-d2_9.csv'
    df = pd.read_csv(f'../Data/{file_name}')
    # meta path independet of the OS
    meta_path = Path('..', 'Data', 'Data_description.xlsx')
    # meta_path = '../Data/Data_description.xlsx'
    meta_df = pd.read_excel(meta_path)
    # extract ppm lines and names
    positions, names = extract_ppm_all(meta_df, file_name)
    
    y_fits = np.zeros((df.shape[0], df.shape[1]))
    fit_params = np.zeros((df.shape[1], 12))
    fit_errors = np.zeros((df.shape[1], 3))

    for i in range(1, df.shape[1]):
        print(f'Fitting column {i/df.shape[1]*100:.2f}%')
        # perform fitting
        x = df.iloc[:,0]
        y = df.iloc[:,i]
        popt, pcov = curve_fit(grey_spectrum, x, y, p0 = [positions, [0.1]*len(positions), [1000]*len(positions)], maxfev=10000)
        y_fits[:,i-1] = grey_spectrum(x, *popt)

        fit_params[i-1] = popt
        #fit_errors[i-1] = np.sqrt(np.diag(pcov))
    #plot_single(x, y_fits, positions, names, file_name, df)
    plot_time_dependece(y_fits, positions, names, fit_params)


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