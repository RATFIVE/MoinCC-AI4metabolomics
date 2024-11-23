#3 subplots for one timeframe, switch timeframes with buttons
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class reference_plots():
    def __init__(self, df):
        self.df =df
        #add variables for lorentzian plot
        self.output = pd.read_csv('./otput/fitting_params.csv')

        def plot(self, i):
            sum_lorentzian = 
            #calculate noise
            y_noise = df.iloc[:,i] - sum_lorentzian


            #create layout
            fig = make_subplots(
                rows=3, 
                cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.02 
            )

            # Spectrum
            fig.add_trace(
                go.Scatter(x=df.iloc[:,0], y=df.iloc[:,i]),
                row=1, col=1
            )

            # Lorentzian curves
            j = 0
            x_values = df.iloc[:,0]

            while True:
                try:
                    width = output.loc[i, f"{j}-width"]
                    amplitude = output.loc[i, f"{j}-amplitude"]
                    position = output.loc[i, f"{j}-position"]
                    
                    y_values = amplitude *width**2 / ((x_values - position)**2 + width**2)

                    fig.add_trace(go.Scatter(x=x_values, y=y_values),
                                  row=2, col=1)
                    
                    j += 1
                except KeyError:
                    break


            # Noise = Spectrum - Lorentzian
            fig.add_trace(
                go.Scatter(),
                row=3, col=1
            )

            # Layout-Anpassungen
            fig.update_layout(height=600, width=600, title_text="Subplots mit geteilter X-Achse")
            fig.show()

