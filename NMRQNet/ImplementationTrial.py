import torch
import torch.nn as nn
import torch.optim as optim
from synthetic_data import generate_synthetic_data

"""
Breakdown of the Model:

Convolutional Layers (CNN): The CNN layers help in extracting spectral features from the NMR input. 
Two convolutional layers with ReLU activation and pooling are used.

Gated Recurrent Unit (GRU): GRU is used to learn positional dependencies between metabolite clusters 
across the chemical shift axis.
Fully Connected Layers: After feature extraction from CNN and sequential learning via GRU, 
fully connected layers are used for final classification or regression (metabolite concentrations).

Loss Function: I used Mean Squared Error (MSE) as the loss function for quantification.
"""

# Define the CNN-GRU architecture
class NMRQNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NMRQNet, self).__init__()
        
        # CNN layers to extract features from the NMR spectra
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # GRU layer to capture sequential information from the extracted features
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(64 * 2, 128)  # 64 * 2 for bidirectional GRU
        self.fc2 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Apply CNN layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        # Reshape for GRU
        x = x.permute(0, 2, 1)  # Rearrange to (batch_size, sequence_length, feature_dim)
        
        # Apply GRU
        x, _ = self.gru(x)
        
        # Take the last hidden state
        x = x[:, -1, :]
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Define training function
def train_model(model, train_loader, num_epochs, criterion, optimizer):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom dataset for NMR spectra
class NMRDataset(Dataset):
    def __init__(self, spectra, labels):
        """
        Args:
            spectra (np.array): Array of NMR spectra (num_samples, spectrum_length)
            labels (np.array): Array of labels or concentrations (num_samples, num_metabolites)
        """
        self.spectra = spectra
        self.labels = labels
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        label = self.labels[idx]
        
        # Convert to PyTorch tensors
        spectrum = torch.tensor(spectrum, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return spectrum.unsqueeze(0), label  # Add channel dimension for CNN (batch_size, 1, spectrum_length)




# Example usage
if __name__ == '__main__':

    # Hyperparameters
    input_size = 10000  # Length of NMR spectrum input
    output_size = 9     # Number of metabolites to identify
    learning_rate = 0.001
    num_epochs = 25
    batch_size = 32


    # Example NMR data (You would replace this with your actual data)
    num_samples = 1000
    spectrum_length = 10000
    num_metabolites = 9


    # # Peaks for FA_20231122_2H_yeast_acetone-d6_1.csv
    # peak_list = [2.323, 4.7, 1.201]
    # num_metabolites = len(peak_list)

    #spectra, concentrations = generate_synthetic_data(peak_list, num_samples, spectrum_length, num_metabolites)

    # Randomly generated data for demonstration (replace with actual NMR spectra and labels)
    spectra = np.random.randn(num_samples, spectrum_length)  # Simulated NMR spectra
    labels = np.random.rand(num_samples, num_metabolites)    # Simulated metabolite concentrations

    # Create dataset and DataLoader
    nmr_dataset = NMRDataset(spectra, labels)

    train_loader = DataLoader(nmr_dataset, batch_size=32, shuffle=True)

    # Check the DataLoader
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"Data shape: {data.shape}, Target shape: {target.shape}")
        break  # Just to show one batch

    # Check if GPU is available, otherwise use CPU
    device = 'cpu'


    # Create the model, loss function, and optimizer
    model = NMRQNet(input_size=input_size, output_size=output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, num_epochs, criterion, optimizer)
