"""
PyTorch neural network models for biological data classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm


class BiologicalClassifier(nn.Module):
    """
    Multi-layer neural network for genomic data classification.
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], output_dim=2, dropout=0.3):
        """
        Initialize neural network.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension
        hidden_dims : list
            Hidden layer dimensions
        output_dim : int
            Number of output classes
        dropout : float
            Dropout rate
        """
        super(BiologicalClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class ConvolutionalClassifier(nn.Module):
    """
    Convolutional neural network for genomic data (1D convolutions).
    """
    
    def __init__(self, input_dim, output_dim=2, n_filters=64, kernel_size=3):
        """
        Initialize CNN.
        
        Parameters
        ----------
        input_dim : int
            Input feature dimension
        output_dim : int
            Number of output classes
        n_filters : int
            Number of convolutional filters
        kernel_size : int
            Kernel size for convolutions
        """
        super(ConvolutionalClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened size after conv and pooling
        flattened_size = (input_dim // 4) * (n_filters * 2)
        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        """Forward pass."""
        # Reshape for 1D convolution: (batch, 1, features)
        x = x.unsqueeze(1)
        
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ModelTrainer:
    """
    Trainer for PyTorch models.
    """
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model : nn.Module
            PyTorch model
        device : str
            'cpu' or 'cuda'
        learning_rate : float
            Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the model.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        validation_split : float
            Fraction of data for validation
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Split into train/val
        n_val = int(len(X_train) * validation_split)
        indices = np.random.permutation(len(X_train))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        X_train_split = X_tensor[train_indices]
        y_train_split = y_tensor[train_indices]
        X_val = X_tensor[val_indices]
        y_val = y_tensor[val_indices]
        
        # Create data loader
        dataset = TensorDataset(X_train_split, y_train_split)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in tqdm(range(epochs), desc="Training"):
            # Train
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(dataloader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val)
                val_acc = (val_outputs.argmax(1) == y_val).float().mean()
            
            self.history['val_loss'].append(val_loss.item())
            self.history['val_acc'].append(val_acc.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    def predict(self, X, return_proba=False):
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        return_proba : bool
            Whether to return probabilities
        
        Returns
        -------
        predictions : np.ndarray
            Class predictions or probabilities
        """
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            if return_proba:
                proba = torch.softmax(outputs, dim=1)
                return proba.cpu().numpy()
            else:
                predictions = outputs.argmax(1)
                return predictions.cpu().numpy()
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
