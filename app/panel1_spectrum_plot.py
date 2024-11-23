import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from LoadData import *
#from Layout import StreamlitApp
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
        title='Spectrum',
        xaxis_title='X',
        yaxis_title='Intensity',
        showlegend=False
    )

    return fig

def get_lorentz(df, columns, xmin, xmax, length):
    for col in columns:
        if 'pos' in col:
            pos = df[col]
        if 'width' in col:
            width = df[col]
            
        if 'amp' in col:
            amp = df[col]
            

    # X-Werte für den Plot
    x = np.linspace(xmin, xmax, length)

    # Berechnen der Lorentz-Funktion für jeden Satz von Parametern

    y = lorentzian(x, pos, width, amp)

    return y

    

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
        title='Lorentzians',
        xaxis_title='X',
        yaxis_title='Intensity',
        showlegend=False
    )

    # Plot anzeigen
    #fig.show()
    return fig

def plot_noise(df, lorentz_array, frame):
        # get the x and y of the i-th spectrum
    df = df.iloc[:, [0, frame]]

    # get the x data
    x = df.iloc[:, 0]
    y = np.zeros_like(x) 

    # get the spectrum as y
    raw = df.iloc[:, 1].values
    

    #print(raw)

    y_max_array = np.maximum.reduce(lorentz_array, axis=0)
    #print(y_max)

    noise = raw - y_max_array

    y_min = min(raw)
    y_max = max(raw)

    fig = go.Figure()

    # Rauschkurve hinzufügen
    fig.add_trace(go.Scatter(x=x, y=noise, mode='lines', name='Noise'))
    #fig.add_trace(go.Scatter(x=x, y=raw, mode='lines', name='raw'))
    #fig.add_trace(go.Scatter(x=x, y=y_max_array, mode='lines', name='lorenz'))

    # Layout der Figur anpassen und y-Achsenbereich festlegen
    fig.update_layout(
        title='Noise Plot',
        xaxis_title='X',
        yaxis_title='Noise',
        yaxis=dict(range=[y_min, y_max])  # y-Achsenbereich festlegen
    )
    
    return fig
    
    

    
    

def streamlit(fig1, fig2, fig3):
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.plotly_chart(fig3)


def main():
    frame = 2
    file_name = 'FA_20231109_2H_yeast_Gluc-d2_5.ser.csv'
    data_list = loaddata.load_data_list(file_name)
    df_raw = pd.read_csv(data_list[0])
    xmin = df_raw.iloc[:, 0].min()
    xmax = df_raw.iloc[:, 0].max()
    fig1 = plot_raw(df_raw, frame)
    
    # for plot 2
    fit_params = iterate_directory(
                      target_dirname=file_name + '_output', 
                      target_filename='fitting_params.csv')
    
    df = pd.read_csv(fit_params)


    substrates = loaddata.get_substrate_list(file_name)
    
    print(substrates)

    metabolites = loaddata.get_metabolite_list(file_name)
    print(metabolites)

    fig_lorentz = go.Figure()

    for number in substrates:
        extracted_df = extract_reacsubs_columns(df, number)
        columns = extracted_df.columns

        plot_df = extracted_df.iloc[frame, :]
        fig2 = plot_lorentz(plot_df, fig_lorentz, columns, xmin, xmax)

    for number in metabolites:
        extracted_df = extract_reacsubs_columns(df,number)
        columns = extracted_df.columns

        plot_df = extracted_df.iloc[frame, :]
        fig2 = plot_lorentz(plot_df, fig_lorentz, columns, xmin, xmax)

    # for Noise Plot
    lorentz_list = []
    for number in substrates + metabolites:
        extracted_df = extract_reacsubs_columns(df, number)
        columns = extracted_df.columns
        df_lorentz = extracted_df.iloc[frame, :]
        length = df_raw.shape[0]
        y = get_lorentz(df=df_lorentz, columns=columns, xmin=xmin, xmax=xmax, length=length)
        lorentz_list.append(y)
    lorentz_array = np.vstack(lorentz_list)
    
    fig3 = plot_noise(df_raw, lorentz_array, frame)
    
    


    #streamlit(fig1, fig2, fig3)

    return (fig1, fig2, fig3)
    



    
        
    
            

    

if __name__ == '__main__':
    main()