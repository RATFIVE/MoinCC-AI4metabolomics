import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from LoadData import *


def lorentzian(x, x0, gamma, height):
    return height * (gamma / ((x - x0)**2 + gamma**2))

def iterate_directory(target_dirname, target_filename):
    loaddata = LoadData()

    output_dir = Path(os.path.join(os.getcwd(), 'output_dir'))
    for root, dirs, files in os.walk(output_dir):
        for dir in dirs:
            # Überprüfen, ob das aktuelle Verzeichnis das Zielverzeichnis ist
            if dir == target_dirname:
                full_dir_path = os.path.join(root, dir)
                for root, dirs, files in os.walk(full_dir_path):
                    for file in files:
                        # Überprüfen ob die datei die Target datei ist
                        if file == target_filename:
                            full_file_path = os.path.join(root, file)
                break
    return full_file_path


# Dynamische Extraktion der Spalten
def extract_reacsubs_columns(df, number):
    pos_col = f'ReacSubs_pos_{number}'
    width_col = f'ReacSubs_width_{number}'
    amp_col = f'ReacSubs_amp_{number}'

    # Überprüfen, ob die Spalten im DataFrame vorhanden sind
    if pos_col in df.columns and width_col in df.columns and amp_col in df.columns:
        extracted_df = df[[pos_col, width_col, amp_col]]
        return extracted_df
    else:
        raise ValueError(f"Columns with number {number} not found in DataFrame")



def main():
    fit_params = iterate_directory(
                      target_dirname='FA_20240108_2H_yeast_Nicotinamide-d4 _7.csv_output', 
                      target_filename='fit_params.csv')
    
    df = pd.read_csv(fit_params)
    
    # Beispielaufruf
    number = '9.094' # Have to get the number from DataDescription
    extracted_df = extract_reacsubs_columns(df, number)
    print(extracted_df)

if __name__ == '__main__':
    main()