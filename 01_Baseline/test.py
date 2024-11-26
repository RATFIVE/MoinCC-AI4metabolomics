import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# /home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/Data/FA_20240731_2H_yeast_Fumarate-d2_15_200.ser.csv
# app/output/FA_20240207_2H_yeast_Fumarate-d2_5.csv_output
file_name = 'FA_20240213_2H_yeast_Fumarate-d2_9.csv'
data_fp = f'../Data/{file_name}'
sum_fp = f'/home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/app/output/{file_name}_output/sum_fit.csv'
individual = f'/home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/app/output/{file_name}_output/substance_fits/sum_fit0.csv'
differences = f'/home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/app/output/{file_name}_output/differences.csv'
fitting_params = f'/home/tom-ruge/Schreibtisch/Fachhochschule/Semester_2/Appl_Project_MOIN_CC/MoinCC-AI4metabolomics/app/output/{file_name}_output/fitting_params.csv'

def lorentzian(x, shift, gamma, A):
    return A * gamma / ((x - shift)**2 + gamma**2)

data = pd.read_csv(data_fp)
sum_data = pd.read_csv(sum_fp)
individual_data = pd.read_csv(individual)
differences = pd.read_csv(differences)
fitting_params = pd.read_csv(fitting_params)

x = data.iloc[:, 0]
y = data.iloc[:, 1]
y_sum = sum_data.iloc[:, 1]
y_diff = differences.iloc[:, 1]
plt.plot(x, y, label='Original')
plt.plot(x, y_sum, label='Sum')
plt.plot(x, y_diff, label='Difference')
for i in range(1, individual_data.shape[1]):
    plt.plot(x, individual_data.iloc[:, i], label=individual_data.columns[i])

# plot water
water_pos = fitting_params.loc[0, 'Water_pos_4.7']
water_width = fitting_params.loc[0, 'Water_width_4.7']
water_amp = fitting_params.loc[0, 'Water_amp_4.7']
print(water_pos, water_width, water_amp)

plt.plot(x, lorentzian(x, water_pos, water_width, water_amp), label='Water from fitting params')
plt.legend()
plt.show()