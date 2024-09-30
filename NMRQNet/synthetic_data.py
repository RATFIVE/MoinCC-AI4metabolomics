import numpy as np
import matplotlib.pyplot as plt
"""
Explanation of the Code:
generate_synthetic_spectrum:
Creates an NMR spectrum by generating random Gaussian peaks at different positions within the spectrum.
Adds Gaussian noise to simulate real-world variations in the data.
generate_synthetic_data:
Calls generate_synthetic_spectrum multiple times to create a dataset of spectra.
Randomly assigns concentrations for a set of num_metabolites metabolites.
Output:
spectra: A NumPy array of shape (1000, 10000), representing 1000 synthetic NMR spectra of length 10,000.
concentrations: A NumPy array of shape (1000, 9), representing the concentrations of 9 metabolites for each spectrum.
"""

def generate_synthetic_spectrum(peak_list, spectrum_length=10000, noise_level=0.01):
    """
    Generate a synthetic NMR spectrum with random peaks and added noise.
    
    Args:
        spectrum_length (int): The length of the NMR spectrum.
        num_peaks (int): The number of peaks to generate in the spectrum.
        noise_level (float): The standard deviation of the noise.
    
    Returns:
        np.array: The generated NMR spectrum.
    """
    spectrum = np.zeros(spectrum_length)
    
    # Add random peaks to the spectrum
    # for _ in range(num_peaks):
    #     peak_position = np.random.randint(0, spectrum_length)
    #     peak_width = np.random.randint(50, 200)
    #     peak_height = np.random.rand() * 2  # Random peak height between 0 and 2
        
    #     # Create a Gaussian peak
    #     peak = peak_height * np.exp(-np.linspace(-3, 3, peak_width)**2)
        
    #     # Add the peak to the spectrum
    #     start = max(0, peak_position - peak_width // 2)
    #     end = min(spectrum_length, peak_position + peak_width // 2)
    #     spectrum[start:end] += peak[:end-start]
    

    # Add peak at pos: 
    labels = []
    for peak_pos in peak_list:
        peak_position = int(peak_pos * spectrum_length/10)
        #peak_position = int(peak_pos)
        peak_width = np.random.randint(50, 200)
        peak_height = np.random.rand() * 2
        # create lorentzian peak
        peak = peak_height * (1 / (1 + np.square(np.linspace(-3, 3, peak_width))))
        # peak = peak_height * np.exp(-np.linspace(-3, 3, peak_width)**2)
        start = max(0, peak_position - peak_width // 2)
        end = min(spectrum_length, peak_position + peak_width // 2)
        spectrum[start:end] += peak[:end-start]
        
        # Generate label for the peak position
        label = f"Metabolite_{peak_pos}"
        labels.append(label)

    #random_noise_level = np.random.rand() * 0.01
    noise_level = 0.01
    # Add random noise to the spectrum
    noise = np.random.normal(0, noise_level, spectrum_length)
    spectrum += noise
    
    return spectrum

def generate_synthetic_data(peak_list, num_samples=1000, spectrum_length=10000, num_metabolites=9):
    """
    Generate synthetic NMR spectra and corresponding metabolite concentrations.
    
    Args:
        num_samples (int): Number of synthetic spectra to generate.
        spectrum_length (int): Length of each spectrum.
        num_metabolites (int): Number of metabolites to simulate concentrations for.
    
    Returns:
        tuple: A tuple (spectra, concentrations)
    """
    spectra = []
    concentrations = []
    
    for _ in range(num_samples):
        # Generate a random NMR spectrum
        spectrum = generate_synthetic_spectrum(peak_list, spectrum_length)
        spectra.append(spectrum)
        
        # Generate random metabolite concentrations (between 0 and 1)
        concentration = np.random.rand(num_metabolites)
        concentrations.append(concentration)
    
    return np.array(spectra), np.array(concentrations)


if __name__ == "__main__":
    # Generate synthetic data
    num_samples = 1000
    spectrum_length = 10000


    # Peaks for FA_20231122_2H_yeast_acetone-d6_1.csv
    peak_list = [2.323, 4.7, 1.201]
    num_metabolites = len(peak_list)

    spectra, concentrations = generate_synthetic_data(peak_list, num_samples, spectrum_length, num_metabolites)
    print("Synthetic data generated successfully!")
    print(f'Spectra: \n {spectra}\n')
    print(f'Concentrations: \n {concentrations}\n')
    # Output shapes
    print("Spectra shape:", spectra.shape)  # Should be (1000, 10000)
    print("Concentrations shape:", concentrations.shape)  # Should be (1000, 9)


    # animate the spctra
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    line, = ax.plot(spectra[0])
    ax.set_title("Synthetic NMR Spectrum Animation")
    ax.set_xlabel("Chemical Shift")
    ax.set_ylabel("Intensity")

    def update(frame):
        line.set_ydata(spectra[frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(num_samples), interval=50)
    plt.show()

    # # Plot one example spectrum
    # plt.figure(figsize=(10, 4))
    # plt.plot(spectra[0])
    # plt.title("Synthetic NMR Spectrum Example")
    # plt.xlabel("Chemical Shift")
    # plt.ylabel("Intensity")
    # plt.show()




    # All Peaks 
    peak_pos_list = [2.323, 6.653, 9.094, 8.876, 8.420, 7.772, 2.468, 4.7, 1.201, 4.368, 2.475, 9.031, 8.714, 8.376, 7.659, 1.2261, 1.9775]