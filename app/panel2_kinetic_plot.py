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
        self.kin_fp = Path('output', os.path.basename(self.path) + '_output', 'kinetics.csv')
        self.kin_df = pd.read_csv(self.kin_fp)

    def plot(self):
        fig = px.line(
            self.kin_df,
            x='time step',
            y=self.kin_df.columns[:-1],  # Select all columns except 'time' for y
            labels={'value': 'Value', 'variable': 'Series'},
            title=f"Kinetic Plot of {self.basename}"
        )

        fig.update_layout(
            title=dict(
                text='Kinetic Plot',
                font=dict(size=30)  # Font size for the title
            ),
            xaxis_title='Time step',
            yaxis_title='Intensity',
            showlegend=True,
            font=dict(
                #family="Courier New, monospace",  # Font family
                size=32,                          # Font size
                #color="RebeccaPurple"             # Font color
            ),
            legend=dict(
                x=0.95,
                y=0.9,
                xanchor='center',
                yanchor='middle'
            ),
            yaxis=dict(
                titlefont=dict(size=24),          # Font size for y-axis title
                tickfont=dict(size=18)            # Font size for y-axis ticks
            ),
            xaxis=dict(
                titlefont=dict(size=24),          # Font size for x-axis title
                tickfont=dict(size=18)            # Font size for x-axis ticks
            )
        )
        # Save the fig as pdf
        pio.write_image(fig, f'Kinetic_{self.basename}.pdf', format='pdf')
        return fig
        