import pandas as pd
import plotly.express as px
from pathlib import Path
import os

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
            x='Time step in a.u.',
            y=self.kin_df.columns[:-1],  # Select all columns except 'time' for y
            labels={'value': 'Value', 'variable': 'Series'},
            title=f"Kinetic Plot of {self.basename}"
        )
        return fig
        