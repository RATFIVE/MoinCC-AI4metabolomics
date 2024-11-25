#3 subplots for one timeframe, switch timeframes with buttons
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy.optimize import curve_fit

class ReferencePlot:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)


class FitReference:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

    def lorentzian(self, x, shift, gamma, A):
        return A * gamma / ((x - shift)**2 + gamma**2)
    
    def fit_reference(self):
        self.fitting_results = pd.DataFrame()