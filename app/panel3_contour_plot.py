import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.colors import ListedColormap
import plotly.io as pio
import os

class ContourPlot():
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.basename = os.path.basename(self.file_path)
        self.plot_dir = Path('output', self.basename + '_output', 'plots')
        self.contour_pdf = Path(self.plot_dir, f'Contour_{self.basename}')

        # Assuming `self.df` is a DataFrame and `self.Z` is created from its data (excluding the first column)
        self.Z = self.df.iloc[:, 1:].to_numpy()

        # Generate x and y ranges
        x = np.arange(self.Z.shape[1])  # Columns correspond to x
        y = self.df.iloc[:,0]  # Rows correspond to y
        
        # Create the meshgrid
        self.X, self.Y = np.meshgrid(x, y)
           
    def plot(self, zmin, zmax):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('white')  # Set the figure background to white
        
        # Create a custom colormap based on 'magma'
        cmap = plt.cm.magma
        new_colors = cmap(np.linspace(0, 1, cmap.N))  # Get the colors of 'magma'
        new_colors[0] = np.array([1, 1, 1, 1])       # Set the lowest value (0) to white
        custom_cmap = ListedColormap(new_colors)

        # Plot with the custom colormap
        contour = ax.contourf(
            self.X,
            self.Y,
            self.Z,
            levels=20,
            cmap=custom_cmap,
            vmin=zmin * self.Z.max(),
            vmax=zmax * self.Z.max()
        )
        ax.set_xlabel('Time step')
        ax.set_ylabel('Chemical Shift [ppm]')
        ax.grid(True)

        # Save the figure as a PDF
        self.save_fig(fig, self.contour_pdf)

        return fig
    
    def save_fig(self, fig, name, width=1200, height=800):
        # Konvertieren der Breite und Höhe von Pixel in Zoll (dpi = 300)
        fig.set_size_inches(width / 100, height / 100)
        fig.savefig(f'{name}.pdf', format='pdf')
        fig.savefig(f'{name}.png', format='png')

