import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import shutil
import time
from pathlib import Path
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
import spm1d as spm

class DataParser:
    def __init__(self):
        pass

    def load_data(self)->list:
        """
        Durchsucht das übergeordnete Verzeichnis des aktuellen Arbeitsverzeichnisses rekursiv nach CSV-Dateien und gibt eine Liste der gefundenen Dateipfade zurück.

        Diese Methode durchsucht das übergeordnete Verzeichnis des aktuellen Arbeitsverzeichnisses rekursiv nach Dateien mit der Endung '.csv'. 
        Alle gefundenen Dateipfade werden in einer Liste gesammelt und zurückgegeben.

        Returns:
            list: Eine Liste von Dateipfaden zu den gefundenen CSV-Dateien.
        """

        path_list = []
        cwd = Path(os.getcwd())

        print(f'Working Dir: {cwd}')
        print(f'Path: {cwd.parent}')

        for root, dirs, files in os.walk(cwd.parent):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    path_list.append(file_path)

        return path_list
    
    def load_df(self, path):
        df = pd.read_csv(path, sep=',', encoding='utf-8')

        # Rename the first column to 'Chemical_Shift'
        df.iloc[:, 0].name == 'Chemical_Shift'
        #df.rename(columns={'Unnamed: 0': 'Chemical_Shift'}, inplace=True)
        print(df.head())
        return df
    



    
model = DataParser()
data = model.load_data()
model.load_df(data[1])