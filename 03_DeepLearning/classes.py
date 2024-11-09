import pandas as pd
import os
from pathlib import Path
import math
import numpy as np
from scipy import interpolate



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
        q1 = np.percentile(y, 20)

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
        tresh = 0
        peaks, prob = find_peaks(y, height=tresh)
        return peaks, prob, tresh
    


# ----------------------------------------------------------
"""
Marcos LoadData class
"""
    


class LoadData:
    def __init__(self):
        """
        Initializes the LoadData class.
        
        This class provides methods for loading data files and retrieving
        information related to specific files, including data descriptions
        and substrate lists.
        """
        pass

    def load_data(self, endswith:str):
        """
        Loads all files from the 'Data' directory that have the specified file extension.

        This function searches through the 'Data' directory (one level up from the
        current working directory) for files that match the specified file extension.
        It collects the paths of these files and returns them as a list.

        Example files:
            - 'FA_20231113_2H_yeast_Pyruvate-d3_1.csv'

        Args:
            endswith (str): The file extension to filter by (e.g., '.csv').

        Returns:
            list: A list of file paths in the 'Data' directory that end with the specified extension.
        """
        path_list = []
        cwd = Path(os.getcwd())
        #print(f'Working Dir: {cwd}')
        data_path = os.path.join(cwd.parent, 'Data')

        #print(f'Path: {data_path}')
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(endswith):
                    file_path = os.path.join(root, file)
                    path_list.append(file_path)
        return path_list

    def load_DataDescription(self):
        """
        Loads the 'DataDescription.csv' file as a pandas DataFrame.

        This function reads 'DataDescription.csv' from the 'Data' directory located
        one level up from the current working directory. It returns the file's content
        as a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the data from 'DataDescription.csv'.
        """
        data_description_path = os.path.join(os.getcwd(), '..', 'Data', 'Data_description_main.xlsx')
        data_description = pd.read_excel(data_description_path, engine='openpyxl')
        #display(data_description)
        return data_description

    def get_substrate_list(self, file: str):
        """
        Retrieves substrate information related to the specified file.

        This function loads data from the 'DataDescription.csv' file, filtering
        rows based on the specified file name. It then extracts the substrate's
        chemical shift (ppm) and water chemical shift (ppm) values, returning them
        as a list.

        Args:
            file (str): The name of the file for which to retrieve substrate information.

        Returns:
            list: A list containing the substrate chemical shift (ppm) and water chemical shift (ppm) as floats.
        """
        data_desc = self.load_DataDescription()

        # Filter by file name
        data_desc = data_desc.loc[data_desc['File'] == file].reset_index(drop=True)

        substrat_shift = data_desc.at[0, 'Substrate_ppm']
        substrat_water = data_desc.at[0, 'Water_ppm']
        
        def clean_list(value):
            # Convert value to float
            return float(value)
        

        # Ensure substrat_shift is treated as a list
        if isinstance(substrat_shift, (float, np.float64)):
            substrat_shift = [substrat_shift]
        else:
            substrat_shift = [clean_list(x) for x in substrat_shift.strip().split(',')]
        
        # Ensure substrat_water is treated as a list
        if isinstance(substrat_water, (float, np.float64)):
            substrat_water = [substrat_water]
        else:
            substrat_water = [clean_list(x) for x in list(substrat_water)]

        # Return as list
        substrates = substrat_shift + substrat_water 
   
        return substrates

    def get_metabolite_list(self, file):
        """
        Retrieves a list of metabolite chemical shifts (ppm) associated with the specified file.

        This method loads data from the 'DataDescription.csv' file and filters it by the specified file name.
        It then extracts all columns containing metabolite information (columns with 'Metabolite' and 'ppm' in
        the name) and returns their values as a list of floats, excluding any NaN values.

        Args:
            file (str): The name of the file for which to retrieve metabolite chemical shift information.

        Returns:
            list: A list of metabolite chemical shift values (in ppm) as floats, excluding NaN values.
    """
        
        data_desc = self.load_DataDescription()
        
        # Filter by file name
        data_desc = data_desc.loc[data_desc['File'] == file].reset_index(drop=True)

        # get all cols which contains Metabolite and ppm
        cols = [col for col in data_desc.columns if 'Metabolite' in col and 'ppm' in col]
        metabolites = []
        for col in cols:
            metabolites.append(data_desc.at[0, col])

        # to float
        if isinstance(metabolites[0], str):
            metabolites = [float(metabolite) for metabolite in metabolites[0].strip().split(',')]
        else:
            metabolites = [float(metabolite) for metabolite in metabolites if not math.isnan(metabolite)]
        
        # remove nan values
        metabolites = [metabolite for metabolite in metabolites if not math.isnan(metabolite)]
        
        return metabolites
    

# ----------------------------------------------------------


def interpolate_to_shape(x_original, y_original, spectrum_lenth=3000):
    """_summary_

    # Originaldaten
    y_original = df.iloc[:, 1]
    x_original = df.iloc[:, 0]

    Args:
        df (_type_): _description_
    """


    # Neue x-Werte (stellen Sie sicher, dass diese innerhalb des Bereichs von x_original liegen)
    x_new = np.linspace(x_original.min(), x_original.max(), spectrum_lenth)

    # Interpolierte Daten
    interpolated_data = interpolate.interp1d(x_original, y_original, kind='linear')(x_new)
    
    df = pd.DataFrame({'x': x_new, 'y': interpolated_data})
    return df

def fill_df(df):
    """If Data is not ranging from -2 to 10, fill the data with noise
    
    """
    # renmame the columns
    df.columns = ['x', 'y']
    x = df.loc[:, 'x']
    y = df.loc[:, 'y']



    # Calculate the step size of the x values
    x_diff = np.diff(x)
    step = np.mean(x_diff)
    
    # Take sample range for the noise
    x_range_lower = 20
    x_range_upper = 100

    # get sample data of the noise
    x_sample = x[x_range_lower:x_range_upper]
    y_sample = y[x_range_lower:x_range_upper]

    # get the max and min values of the sample data
    y_min, y_max = y_sample.min(), abs(y_sample.min())
    x_min, x_max = x_sample.min(), abs(x_sample.min())

    if x_min > -2:

        # create values in n steps
        x_new = np.arange(-2, x.iloc[0], step)
        y_new = np.zeros_like(x_new)

        # set the noise level
        noise = np.random.normal(y_min, y_max, len(x_new))
        
        # smooth the noise with gaussian filter
        # Berechne die Standardabweichung der y-Daten
        sigma = len(y_new) / 100
        noise = gaussian_filter1d(input=noise, sigma=sigma)

        # replace y_new with noise
        y_new = noise

        data = pd.DataFrame({'x': x_new, 'y': y_new})
        df = pd.concat([data, df], axis=0)
        df.reset_index(drop=True, inplace=True)

    if x_max < 10:
    
        # create values in n steps
        x_new = np.arange(x.iloc[-1], 10, step)
        y_new = np.zeros_like(x_new)

        # set the noise level
        noise = np.random.normal(y_min, y_max, len(x_new))

        # smooth the noise with gaussian filter
        # Berechne die Standardabweichung der y-Daten
        sigma = len(y_new) / 100
        noise = gaussian_filter1d(input=noise, sigma=sigma)

        # replace y_new with noise
        y_new = noise

        data = pd.DataFrame({'x': x_new, 'y': y_new})
        df = pd.concat([df, data], axis=0)
        df.reset_index(drop=True, inplace=True)
                
    return df