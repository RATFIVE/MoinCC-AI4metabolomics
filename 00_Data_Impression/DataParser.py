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
        print(f'Path: {cwd.parent}')

        for root, dirs, files in os.walk(cwd.parent):
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
    

if __name__ == '__main__':

    model = DataParser()
    data = model.load_data()
    model.load_df(data[1])