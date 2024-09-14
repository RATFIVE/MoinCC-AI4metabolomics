import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import shutil
import time
from pathlib import Path

# Options to display all columns
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

def load_data():
    path = Path('Data')
    path_list = []
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                path_list.append(file_path)
            
    
    return path_list


def load_df(path):
    df = pd.read_csv(path)
    return df

def df_description(df):
    print(f'Dataframe Shape: {df.shape}')
    print(f'DataFrame Head: \n{df.head()}')
    #print(f'Dataframe Description: \n{df.describe()}')

    

    return None

def main():
    path_list = load_data()
    df = load_df(path_list[1])
    df_description(df)
    






if __name__ == '__main__':
    main()