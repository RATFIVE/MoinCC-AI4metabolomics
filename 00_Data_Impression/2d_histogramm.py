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

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_2d_hist(df, positions, names, output_direc):
    # Create the figure
    plt.figure(figsize=(10, 10))

    # Extract the data for the heatmap
    data_matrix = df.iloc[:, 1:].values

    # Create the heatmap
    plt.pcolormesh(data_matrix, cmap='hot', shading='auto')

    # Set x-ticks (ppm)
    plt.xticks(np.arange(0.5, len(positions), 1), positions, rotation=90)

    # Set y-ticks, display every Nth tick for clarity
    step_size = np.round(max(1, len(df) // 10))  # Adjust step size as needed
    y_tick_positions = np.arange(0, len(df), step_size)  # Generate ticks at intervals
    y_tick_labels = df.iloc[y_tick_positions, 0]  # Use the first column as the label
    plt.yticks(y_tick_positions + 0.5, y_tick_labels)  # +0.5 aligns ticks with cells

    # Add labels and colorbar
    plt.ylabel('ppm')
    plt.xlabel('Time')
    plt.colorbar()

    # Add title
    plt.title(f'2D Histogram of {os.path.basename(output_direc)[:-4]}')

    # Add horizontal ppm lines
    for i, pos in enumerate(zip(positions, names)):
        plt.axhline(i, color='blue', linestyle='--')
        plt.text(len(df) + 1, i, f'{pos[1]}: {pos[0]} ppm', color='blue')

    # Save and display the plot
    os.makedirs(os.path.dirname(output_direc), exist_ok=True)
    plt.savefig(f'{output_direc}.png')
    plt.close()



def main():
    file_names  = get_file_names()
    for file_name in file_names:
        print(f'Processing {file_name}')
        df = pd.read_csv(f'../Data/{ file_name}')
        # meta path independet of the OS
        meta_path = Path('..', 'Data', 'Data_description.xlsx')
        # meta_path = '../Data/Data_description.xlsx'
        meta_df = pd.read_excel(meta_path)
        # extract ppm lines and names
        positions, names = extract_ppm_all(meta_df, file_name)
        output_direc = f'output/2d_hist/{file_name}_output'
        plot_2d_hist(df, positions, names, output_direc)



main()
