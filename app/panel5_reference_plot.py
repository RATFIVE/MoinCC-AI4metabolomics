import pandas as pd
import matplotlib.pyplot as plt
import re
import peak_fitting_v6



class Reference():
    def __init__(self, fp_file, fp_meta):
        self.data = pd.read_csv(fp_file)
        self.chem_shifts = self.data.iloc[:,0]
        self.LorentzianFit = peak_fitting_v6.PeakFitting(fp_file = fp_file , fp_meta = fp_meta)
        self.fitting_params = self.LorentzianFit.fit(save_csv= False)
        self.reference_value = self.ReferenceValue()
    
    def ReferenceValue(self):
        #get referenz concentration from meta data
        mmol = re.findall(r'[0-9\.]+', self.LorentzianFit.meta_df.iloc[0]['Substrate_mM_added'])
        mmol = float(mmol[0])

        if mmol:
            print(mmol)  
        else:
            print("mMol value couldn't be extracted from Substrate_mM_added ")

        #calculate ref_factor
        ref_factor = mmol / self.fitting_params['Water_amp_4.7'].mean()

        return ref_factor

    # def plot_water_amplitude(self):

    #     fig, ax = plt.subplots()
    #     ax.plot(self.fitting_params['Water_amp_4.7'])
    #     ax.set_xlabel('time')  
    #     ax.set_ylabel('integral value water peak')
        
    #     plt.show()
        
    #     return fig #, ax  
     
    
    # def plot_lorentzian(self, i):
        
    #     spectra_data = self.data.iloc[:,i+1]
        
    #     fig, ax = plt.subplots()
        
    #     # actual curve
    #     ax.plot(self.chem_shifts, spectra_data, c='blue', label='Reference Spectrum')
        
    #     # Lorentzian
    #     y_lorentzian = self.LorentzianFit.lorentzian(x=self.data.iloc[:,0], 
    #                                              shift= self.fitting_params.iloc[i]['Water_pos_4.7'],
    #                                              gamma= self.fitting_params.iloc[i]['Water_width_4.7'], 
    #                                              A= self.fitting_params.iloc[i]['Water_amp_4.7'])
    #     ax.plot(self.chem_shifts, y_lorentzian, c='red', label='Lorentzian fit')
        
    #     ax.set_xlabel('Chemical Shifts')
    #     ax.set_ylabel('Intensity')
    #     ax.legend()
        
    #     plt.show()
        
    #     return fig #, ax 
    
    def plot(self, i):
        spectra_data = self.data.iloc[:,i+1]


        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # amplitude
        ax[0].plot(self.fitting_params['Water_amp_4.7'])
        ax[0].axhline(y=self.fitting_params['Water_amp_4.7'].mean(), color='grey', linestyle='--')
        ax[0].set_title('Integral of water over time')
        ax[0].set_xlabel('Time step') 
        ax[0].set_ylabel('Integral value water peak')
        # annotation
        ax[0].annotate(f'Calculated Convergence Factor = {self.reference_value:.3f}', 
                       xy=(1.05, 0.85), xycoords='axes fraction', 
                       xytext=(-20, 20), 
                       textcoords='offset points',
                       fontsize = 8, 
                       ha='right', 
                       va='top')

        #axs[0].legend()

        # Second plot
        #actual curve
        ax[1].plot(self.chem_shifts, spectra_data, c='blue', label='Reference spectrum')
        
        # Lorentzian
        y_lorentzian = self.LorentzianFit.lorentzian(x=self.data.iloc[:,0], 
                                                 shift= self.fitting_params.iloc[i]['Water_pos_4.7'],
                                                 gamma= self.fitting_params.iloc[i]['Water_width_4.7'], 
                                                 A= self.fitting_params.iloc[i]['Water_amp_4.7'])
        ax[1].plot(self.chem_shifts, y_lorentzian + self.fitting_params.iloc[i]['y_shift'] , c='red', label='Lorentzian fit')
        
        
        ax[1].set_xlabel('Chemical shift [ppm]')
        ax[1].set_ylabel('Intensity')
        ax[1].set_title(f'Lorentzian fit for time step: {i}')
        ax[1].set_xlim(max(self.chem_shifts),min(self.chem_shifts))
        ax[1].legend()
        plt.tight_layout()
        
    
        return fig   
    
