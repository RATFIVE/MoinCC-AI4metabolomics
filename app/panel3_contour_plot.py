import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

class ContourPlot():
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

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
        return fig

