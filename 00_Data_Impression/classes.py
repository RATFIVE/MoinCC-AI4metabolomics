import pandas as pd
import os
from pathlib import Path


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
        #print(f'Path: {cwd.parent}')
        data_path = Path(cwd.parent, 'Data')
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    path_list.append(file_path)

        return path_list
    
    def load_df(self, path):
        """
        Lädt eine CSV-Datei in ein Pandas DataFrame und benennt die erste Spalte in 'Chemical_Shift' um.

        Diese Methode lädt eine CSV-Datei von dem angegebenen Pfad in ein Pandas DataFrame. 
        Die erste Spalte der CSV-Datei, die normalerweise keinen Namen hat und als 'Unnamed: 0' bezeichnet wird, 
        wird in 'Chemical_Shift' umbenannt.

        Args:
            path (str): Der Pfad zur CSV-Datei, die geladen werden soll.

        Returns:
            pd.DataFrame: Ein Pandas DataFrame, das die Daten der CSV-Datei enthält, wobei die erste Spalte in 'Chemical_Shift' umbenannt wurde.
        """
        
        df = pd.read_csv(path, sep=',', encoding='utf-8')

        # Rename the first column to 'Chemical_Shift'
        df.rename(columns={'Unnamed: 0': 'Chemical_Shift'}, inplace=True)
        
        return df
    








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









import pandas as pd
import numpy as np
from scipy.signal import find_peaks


class PeakFinder:
    def __init__(self):
        pass

    def threshold(self, y: np.ndarray) -> float:
        """
        Berechnet einen Schwellenwert basierend auf dem Mittelwert und der Standardabweichung eines gegebenen Arrays.

        Diese Methode berechnet den Schwellenwert als Summe aus dem Mittelwert und der Standardabweichung der Werte im Array `y`.

        Args:
            y (np.ndarray): Ein Array von numerischen Werten, für das der Schwellenwert berechnet werden soll.

        Returns:
            float: Der berechnete Schwellenwert, der die Summe aus dem Mittelwert und der Standardabweichung des Arrays `y` darstellt.
        """
        mean = np.mean(y, axis=0)
        std = np.std(y)
        q1 = np.percentile(y, 25)

        tresh = q1 
    
        #print(f'Mean: {mean}')
        return tresh 
    
    def peaks_finder(self, y: np.ndarray) -> tuple:
        """
        Findet Peaks in den Spektren basierend auf den y-Werten und einem berechneten Schwellenwert.

        Diese Methode verwendet die y-Werte der Spektren, um Peaks zu identifizieren. 
        Der Schwellenwert für die Peak-Erkennung wird als Summe aus dem Mittelwert und der Standardabweichung der y-Werte berechnet.

        Args:
            y (np.ndarray): Ein Array von y-Werten der Spektren.

        Returns:
            tuple: Ein Tupel bestehend aus:
                - peaks (np.ndarray): Indizes der gefundenen Peaks.
                - prob (dict): Ein Dictionary mit zusätzlichen Informationen über die Peaks, wie z.B. deren Höhe.
        """
        # Finde Peaks in y 
        tresh = self.threshold(y)
        peaks, prob = find_peaks(y, height=tresh)
        return peaks, prob, tresh