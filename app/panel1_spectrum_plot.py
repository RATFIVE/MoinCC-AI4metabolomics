import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from LoadData import *
from Layout import StreamlitApp
import streamlit as st

loaddata = LoadData()


# Lorentz-Funktion definieren
def lorentzian(x, pos, width, amp):
    return amp * (width / ((x - pos)**2 + width**2))

def iterate_directory(target_dirname, target_filename):
    

    output_dir = Path(os.path.join(os.getcwd(), 'output'))
    
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
def extract_reacsubs_columns(df,number):
    pos_col = f'pos_{number}'
    width_col = f'width_{number}'
    amp_col = f'amp_{number}'

    # Überprüfen, ob die Spalten im DataFrame vorhanden sind
    for col in df.columns:
        if pos_col in col:
            pos_col = col
        if width_col in col:
            width_col = col
        if amp_col in col:
            amp_col = col
           
    extracted_df = df[[pos_col, width_col, amp_col]]
    
    return extracted_df
 
def plot_raw(df, i):

    # get the x and y of the i-th spectrum
    df = df.iloc[:, [0, i]]

    # get the x data
    x = df.iloc[:, 0]

    # get the spectrum as y
    y = df.iloc[:, 1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Raw'))

    fig.update_layout(
        title='Lorentzian Plot',
        xaxis_title='X',
        yaxis_title='Intensity',
        showlegend=False
    )

    return fig

    

def plot_lorentz(df, fig, columns, xmin, xmax):
    for col in columns:
        if 'pos' in col:
            pos = df[col]
        if 'width' in col:
            width = df[col]
            
        if 'amp' in col:
            amp = df[col]
            

    # X-Werte für den Plot
    x = np.linspace(xmin, xmax, 1000)

    # Berechnen der Lorentz-Funktion für jeden Satz von Parametern

    y = lorentzian(x, pos, width, amp)

    

    # Lorentz-Kurve hinzufügen
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Lorentzian'))

    # Layout der Figur anpassen
    fig.update_layout(
        title='Lorentzian Plot',
        xaxis_title='X',
        yaxis_title='Intensity',
        showlegend=False
    )

    # Plot anzeigen
    #fig.show()
    return fig

def streamlit(fig1, fig2):
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)


def main():
    frame = 10
    file_name = 'FA_20231109_2H_yeast_Gluc-d2_5.ser.csv'
    data_list = loaddata.load_data_list(file_name)
    df_raw = pd.read_csv(data_list[0])
    xmin = df_raw.iloc[:, 0].min()
    xmax = df_raw.iloc[:, 0].max()
    fig1 = plot_raw(df_raw, frame)
    

    fit_params = iterate_directory(
                      target_dirname=file_name + '_output', 
                      target_filename='fitting_params.csv')
    
    df = pd.read_csv(fit_params)
    column_set = df.columns
    column_set = [col.split('_') for col in column_set]
    
    # Bereinigen der Liste
    column_set = [col[0] for col in column_set if col[0] != 'Time']

    column_set = set(column_set)


    # Beispielaufruf
    numbers = ['3.844', '3.975'] # Have to get the number from DataDescription
    substrates = loaddata.get_substrate_list(file_name)
    
    print(substrates)

    metabolites = loaddata.get_metabolite_list(file_name)
    print(metabolites)

    fig_lorentz = go.Figure()

    for number in substrates:
        extracted_df = extract_reacsubs_columns(df,number)
        columns = extracted_df.columns

        plot_df = extracted_df.iloc[frame, :]
        fig2 = plot_lorentz(plot_df, fig_lorentz, columns, xmin, xmax)

    for number in metabolites:
        extracted_df = extract_reacsubs_columns(df,number)
        columns = extracted_df.columns

        plot_df = extracted_df.iloc[frame, :]
        fig2 = plot_lorentz(plot_df, fig_lorentz, columns, xmin, xmax)

    #streamlit(fig1, fig2)

    return (fig1, fig2)
    



    
        
    
            

    

if __name__ == '__main__':
    main()