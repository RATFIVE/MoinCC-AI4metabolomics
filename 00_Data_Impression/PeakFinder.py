import pandas as pd
import numpy as np
from scipy.signal import find_peaks




class PeakFinder:
    def __init__(self):
        pass

    def threshold(self, y: np.ndarray) -> float:
        """
        Berechnet einen Schwellenwert basierend auf dem Mittelwert und der Standardabweichung eines gegebenen Arrays.

        Diese Methode berechnet den Schwellenwert als Summe aus dem Mittelwert und der Standardabweichung der Werte im Array `y`.

        Args:
            y (np.ndarray): Ein Array von numerischen Werten, für das der Schwellenwert berechnet werden soll.

        Returns:
            float: Der berechnete Schwellenwert, der die Summe aus dem Mittelwert und der Standardabweichung des Arrays `y` darstellt.
        """
        mean = np.mean(y, axis=0)
        std = np.std(y)

        tresh = mean 
    
        #print(f'Mean: {mean}')
        return tresh
    
    def peaks_finder(self, y: np.ndarray) -> tuple:
        """
        Findet Peaks in den Spektren basierend auf den y-Werten und einem berechneten Schwellenwert.

        Diese Methode verwendet die y-Werte der Spektren, um Peaks zu identifizieren. 
        Der Schwellenwert für die Peak-Erkennung wird als Summe aus dem Mittelwert und der Standardabweichung der y-Werte berechnet.

        Args:
            y (np.ndarray): Ein Array von y-Werten der Spektren.

        Returns:
            tuple: Ein Tupel bestehend aus:
                - peaks (np.ndarray): Indizes der gefundenen Peaks.
                - prob (dict): Ein Dictionary mit zusätzlichen Informationen über die Peaks, wie z.B. deren Höhe.
        """
        # Finde Peaks in y 
        tresh = self.threshold(y)
        peaks, prob = find_peaks(y, height=tresh)
        return peaks, prob