import pandas as pd
import matplotlib.pyplot as plt


input_file = '/home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/Data/FA_20240108_2H_yeast_Nicotinamide-d4 _11.csv'
meta_file = '/home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/Data/AI4metabolomics_Project_Description.docx'


# get file names
def get_file_names():
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
            path_list.append(file)
    return path_list


def containing_string(file_names, string = '', not_string = None):
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
    return [file for file in file_names if string in file and (not_string is None or not_string not in file)]

def plot_integrated_peaks(fp):
    df = pd.read_csv(fp)
    for col in df.columns[1:]:
        plt.plot(df['Time'], df[col], label=col)
    
    plt.show()
