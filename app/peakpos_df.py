import pandas as pd
import numpy as np


class SpectraAnalysis:
    def __init__(self):
        pass

    def peakfit_sum(self, spectra_data, chem_shifts, threshold):
        """Summen der Spektren und Rückgabe einer Liste gefundener Peaks"""
        sum_of_spectra = np.sum(spectra_data, axis=1)
        threshold = np.percentile(sum_of_spectra, threshold)

        first_derivative = np.gradient(sum_of_spectra, chem_shifts)
        second_derivative = np.gradient(first_derivative, chem_shifts)
        third_derivative = np.gradient(second_derivative, chem_shifts)
        fourth_derivative = np.gradient(third_derivative, chem_shifts)

        sign_change = np.diff(np.sign(third_derivative)) != 0
        peak_mask = (sum_of_spectra > threshold) & (second_derivative < 0) & (fourth_derivative > 0)
        peak_mask[1:] &= sign_change

        peak_pos = chem_shifts[peak_mask].tolist()
        return peak_pos

    def normalize_water(self, data):
        """Anpassung der Daten basierend auf dem Wasser-Peak"""
        spectra_data = data.iloc[:, 1:]
        chem_shifts = data.iloc[:, 0]

        peak_pos = self.peakfit_sum(spectra_data, chem_shifts, 85)
        water = 4.7
        closest_peak = min(peak_pos, key=lambda x: abs(x - water))
        
        data_normalized = pd.DataFrame(chem_shifts.copy() + (4.7 - closest_peak))
        data_normalized = pd.concat([data_normalized, data.iloc[:, 1:]], axis=1)
        data_normalized.columns = data.columns
        return data_normalized

    def peak_identify(self, data_normalized, expected_peaks, initial_threshold=85, max_shift=0.5): #max_shift maybe
        """Identifizierung von Peaks und Zuordnung zu erwarteten Peaks"""
        spectra_data = data_normalized.iloc[:, 1:]
        chem_shifts = data_normalized.iloc[:, 0]

        found = [None] * len(expected_peaks)
        other = []
        threshold = initial_threshold

        while None in found or len(other) <= len(found):
            detected_peaks = self.peakfit_sum(spectra_data, chem_shifts, threshold)
            for peak in detected_peaks:
                distances = [abs(peak - expected_peak) for expected_peak in expected_peaks]
                min_distance = min(distances)
                index = distances.index(min_distance)

                if min_distance <= max_shift:
                    if found[index] is None:
                        found[index] = peak
                    elif found[index] != peak:
                        other.append(peak)
                else:
                    other.append(peak)

            threshold -= 2
            if threshold < 0 or None not in found or len(other) > len(found):
                break

        print("Found Peaks: ", found)
        print("Other Peaks: ", other)

        return found, other

    def peak_df(self, data, expected_peaks, min_cols_per_section=20):
        """Normalizes the given data to chem_shift(water) = 4.7 
        splits spectra into sections over time
        identifies peaks in summed up sections

        Args:
            data (dataFrame): spectrum data; 1.column with chemical shift, athers with intensity
            expected_peaks (list): list of chem_shifts of expected peaks
            min_cols_per_section (int, optional): minimum number if spectra that are summed up to find peaks. Defaults to 20.

        Returns:
            df: Dtaframe with found peaks
        """
        #normalize data
        df = self.normalize_water(data)
    
        # Split into section with at least 20 columns each
        cols_to_divide = len(df.columns)-1 #-1 because first column is chemical shift
        num_sections = max(cols_to_divide // min_cols_per_section, 1) if cols_to_divide > min_cols_per_section else 1

        cols_per_section = cols_to_divide // num_sections
        extra_cols = cols_to_divide % num_sections  # Extra columns to distribute

        sections = []
        start_col = 1 #ignore first column
        col_sect = []

        for section in range(num_sections):
            extra = 1 if section < extra_cols else 0
            end_col = start_col + cols_per_section + extra
            #sections.append(df.iloc[:, start_col:end_col])
            sections.append(df.iloc[:, [0] + list(range(start_col, end_col))]) #add column with chem_shift to each section
            start_col = end_col
            col_sect.extend([f"{section + 1}"] * (cols_per_section + extra)) #column entries for final dataframe

        found_lists = []
        other_lists = []

        # Find peaks for each section
        for df_section in sections:
            found, other = self.peak_identify(df_section, expected_peaks, compound_names)
            found_lists.append(found)
            other_lists.append(other)

        
        #create final dataframe
        final_df = pd.DataFrame({
            'Column': df.columns[1:],  # Exclude the chemical shift from the final DataFrame columns
            'section': col_sect
        })


        peaks_data = {'Found Peaks': [], 'Other Peaks': []}
        for i in range(num_sections):
            section_length = len(sections[i].columns) - 1  # -1 to ignore the chemical shift column
            peaks_data['Found Peaks'].extend([found_lists[i]] * section_length)
            peaks_data['Other Peaks'].extend([other_lists[i]] * section_length)

        final_df['Found Peaks'] = peaks_data['Found Peaks']
        final_df['Other Peaks'] = peaks_data['Other Peaks']

        return final_df
