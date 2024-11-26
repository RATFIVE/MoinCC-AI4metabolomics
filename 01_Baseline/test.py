import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fp = 'output/FA_20240517_2H_yeast_Nicotinamide-d4 _3.csv_output/fitting_params.csv'
# /home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/Data/FA_20240731_2H_yeast_Fumarate-d2_15_200.ser.csv
# Data/FA_20240207_2H_yeast_Fumarate-d2_5.csv
data_fp = '../Data/FA_20240517_2H_yeast_Nicotinamide-d4 _3.csv'

data = pd.read_csv(data_fp)
def lorentzian(x, shift, gamma, A):
    return A * gamma / ((x - shift)**2 + gamma**2)

df = pd.read_csv(fp)
print(df)

reac_subs_pos = df['Water_pos_4.7'].to_numpy()
reac_subs_width = df['Water_width_4.7'].to_numpy()
reac_subs_amp = df['Water_amp_4.7'].to_numpy()

x = np.linspace(1, 12, 1000)
y = lorentzian(x, reac_subs_pos[1], reac_subs_width[1], reac_subs_amp[1])

plt.plot(x, y)
plt.plot(data.iloc[:,0], data.iloc[:,2])
plt.show()


