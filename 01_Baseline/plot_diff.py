import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV file
fp = 'app/output/FA_20240108_2H_yeast_Nicotinamide-d4 _11.csv_output/differences.csv'

# Load the data
df = pd.read_csv(fp)

# Iterate through columns starting from the second one
for col in df.columns[1:]:
    plt.plot(df.iloc[:, 0], df[col], label=col)
    plt.xlabel(df.columns[0])  # Label x-axis with the first column name
    plt.ylabel(col)  # Label y-axis with the current column name
    plt.title(f"Plot for {col}")
    plt.legend()
    plt.show()
    
