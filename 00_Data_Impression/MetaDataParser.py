import pandas as pd
import numpy as np
import os

"""
Substrate:  Substrate_name      ->      Substrate_chemical shift (ppm)
            Metabolite_1        ->      Metabolite_1_ppm
            Metabolite_2        ->      Metabolite_2_ppm


"""

class MetaDataParser:
    def __init__(self):
        pass

    def load_data_desciption(self, path:str)->pd.DataFrame:
        """

        """
        df = pd.read_excel(path, engine='openpyxl')
        return df
    
    def load_substrate(self, path):
        """
        Lädt die Substrate aus der Metadaten-Datei und gibt sie als Liste zurück.

        Diese Methode lädt die Substrate aus der Metadaten-Datei und gibt sie als Liste zurück.

        Returns:
            list: Eine Liste der Substrate, die in der Metadaten-Datei aufgeführt sind.
        """
        
        df = self.load_data_desciption(path)
        metabolite_dict = {'Substrate_name': 'Substrate_chemical shift (ppm)',
                      'Metabolite_1': 'Metabolite_1_ppm',
                      'Metabolite_2': 'Metabolite_2_ppm'}
        
        
        # Auswahl der Spalten basierend auf den Schlüsseln und Werten in metabolite_dict
        selected_columns = ['File'] + list(metabolite_dict.keys()) + list(metabolite_dict.values()) 
        selected_df = df[selected_columns]

        print(selected_df)



if __name__ == '__main__':
    path = '/Users/marco/Documents/MoinCC-AI4metabolomics/Data/Data_description.xlsx'
    model = MetaDataParser()
    df = model.load_data_desciption(path)
    model.load_substrate(path)