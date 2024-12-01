import pandas as pd
import plotly.express as px
from pathlib import Path
import os
import plotly.io as pio

class KineticPlot:
    def __init__(self, path):   
        self.path = path
        # init path
        self.basename = os.path.basename(self.path)
        self.plot_dir = Path('output', self.basename + '_output', 'plots')
        self.kin_fp = Path('output', os.path.basename(self.path) + '_output', 'kinetics.csv')
        self.kin_df = pd.read_csv(self.kin_fp)
        self.kinetic_pdf = Path(self.plot_dir, f'Kinetic_{self.basename}')
        self.colors = px.colors.qualitative.Alphabet
        self.template = 'plotly_white'
 
    def plot(self):
        colors = self.colors[3:]
        fig = px.scatter(
            self.kin_df,
            x='Time_Step',
            y=self.kin_df.columns[1:],  # Select all columns except 'time' for y
            labels={'value': 'Value', 'variable': 'Series'},
            title=f"Kinetic Plot of {self.basename}",
            color_discrete_sequence=colors
        )

        fig.update_layout(
            title='Kinetic Plot',
            xaxis_title='Time step',
            yaxis_title='Intensity',
            showlegend=True,
            font=dict(
                size=20,                          # Font size
            ), 
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            template=self.template
            )
    
        # Save the fig as pdf
        self.save_fig(fig, self.kinetic_pdf)
        
        return fig
    
    def save_fig(self, fig, name):
        pio.write_image(fig, f'{name}.pdf', format='pdf', engine='kaleido', width=1200, height=800)
        pio.write_image(fig, f'{name}.png', format='png', engine='kaleido', width=1200, height=800) 
        