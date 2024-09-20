import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


# read metadata
pd.set_option('display.max_columns', None)

# get current working directory
cwd = os.getcwd()
# meta path independet of the OS
meta_path = Path('..', 'Data', 'Data_description.xlsx')
#meta_path = '../Data/Data_description.xlsx'
meta_df = pd.read_excel(meta_path)

# marcos load data
def get_file_names():
    path_list = []
    # data_direc independent of the OS
    data_direc = Path('..','Data')
    # get all filenames which end with .csv
    for file in os.listdir(data_direc):
        if file.endswith('.csv'):
            path_list.append(file)
    return path_list

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

    # maximum height
    max_height = np.max(df[:, 1:])
    plt.ylim(0, max_height + 1000)

    min_height = np.min(df[:, 1:]) 
    min_height += 1/12 * (max_height - min_height)

    # make vertical lines for each ppm, i want that to be at the first quartile
    height = min_height
    for i in range(len(ppm_lines)):
        plt.axvline(x=ppm_lines[i], color='r', linestyle='--', label='ppm')
        plt.text(ppm_lines[i], height, names[i], rotation=0)
        height += 1/12 * (max_height - min_height)

    # if output dir does not exist, create it
    if not os.path.exists('output'):
        os.makedirs('output')
    # save the plot
    plt.savefig(f'output/{file_name}_plot_{y}.png')
    # close figure
    plt.close()

from PIL import Image
import glob

# Create a GIF from the saved plots
def create_gif(file_name, output_dir='output', gif_name='nmr_spectrum.gif'):
    # Find all saved plot images (e.g., output/file_name_plot_1.png, output/file_name_plot_2.png, ...)
    image_files = glob.glob(f"{output_dir}/{file_name}_plot_*.png")

    # sort images by last number in file name
    image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
    # Open each image
    images = [Image.open(img) for img in image_files]
    
    # Save as GIF
    images[0].save(f"{output_dir}/{gif_name}", save_all=True, append_images=images[1:], duration=100, loop=0)
    
    print(f"GIF saved as {output_dir}/{gif_name}")

def main():
    file_names = get_file_names()
    for file_name in file_names:
        print(f'Processing {file_name}')
        df = pd.read_csv(f'../Data/{file_name}')
        df = df.to_numpy()
        ppm_lines, names = extract_ppm_all(meta_df, file_name)
        for y in range(1, df.shape[1]):
            if ppm_lines == []:
                continue
            single_plot(df, y, ppm_lines, names, file_name)
        if ppm_lines != []:
            create_gif(file_name, output_dir='output', gif_name=f'nmr_spectrum_{file_name}.gif')

        # remove all png files
        for file in os.listdir('output'):
            if file.endswith('.png'):
                os.remove(f'output/{file}') 
        
main()