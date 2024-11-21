import pandas as pd
import matplotlib.pyplot as plt
from peak_fitting_v6 import PeakFitting
import os
from pathlib import Path


meta_file = '/home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/Data/Data_description_main.xlsx'

class ProcessWrite:
    def __init__(self, meta_file, file_name):
        self.meta_file = meta_file
        self.file_names = self.get_file_names()
        self.file_names = self.containing_string(file_name)
        
    # get file names
    def get_file_names(self):
        """
        Get all filenames in the data directory. Using the Path library to make the code OS independent. Files need to end with .csv

        Returns:
            path_list: list of all filenames in the data directory
        """
        path_list = []
        # data_direc independent of the OS
        data_direc = Path('..','Data')
        # get all filenames which end with .csv
        for file in os.listdir(data_direc):
            if file.endswith('.csv'):
                path_list.append(str(data_direc)+'/'+ file)
        return path_list


    def containing_string(self,  string = '', not_string = None):
        """
        Get all filenames which contain a specific string. If not_string is given, the string must be present and the not_string must not be present.

        Args:
            file_names: list of all filenames
            string: string which should be present in the filename
            not_string: string which should not be present in the filename

        Returns:
            list: list of all filenames which contain the string
        """
        # get all filenames which contain the string
        return [file for file in self.file_names if string in file and (not_string is None or not_string not in file)]
    
    def fit_all_files(self):
        """
        Fit all files in the data directory. The files are fitted with the PeakFitting class and the results are written to a new csv file.
        """
        # iterate over all files
        for file in self.file_names:
            # create the PeakFitting object
            peak_fitting = PeakFitting(file, self.meta_file)
            # fit the data
            peak_fitting.fit()
            # write the results to a new csv file
            peak_fitting.write_results()

runner = ProcessWrite(meta_file, 'Acetone')
runner.fit_all_files()

class PlotResults:
    def __init__(self, meta_file, file_name):
        self.meta_file = meta_file

    def lorentzian(self, x, x0, gamma, A):
        return A / (1 + ((x - x0) / gamma) ** 2)
    
    