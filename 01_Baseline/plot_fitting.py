import pandas as pd
import matplotlib.pyplot as plt


fp = '/home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/01_Baseline/output_dir/FA_20240108_2H_yeast_Nicotinamide-d4 _7.csv_output/integrated_peaks_error.csv'
def plot_integrated_peaks(fp):
    df = pd.read_csv(fp)
    for col in df.columns[1:]:
        plt.plot(df['Time'], df[col], label=col)
    
    plt.show()
    
plot_integrated_peaks(fp)
