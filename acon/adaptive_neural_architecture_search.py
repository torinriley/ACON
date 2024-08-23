import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ANASearchSpace:
    def __init__(self):
        # Define the search space
        self.search_space = {
            'num_layers': [2, 3, 4],
            'layer_type': ['dense', 'conv'],
            'num_units': [32, 64, 128],
            'activation': [nn.ReLU, nn.Tanh, nn.Sigmoid],
            'optimizer': ['adam', 'sgd'],
        }

    def sample_architecture(self):
        # Sample a random architecture from the search space
        architecture = {
            'num_layers': np.random.choice(self.search_space['num_layers']),
            'layer_type': np.random.choice(self.search_space['layer_type']),
            'num_units': np.random.choice(self.search_space['num_units']),
            'activation': np.random.choice(self.search_space['activation']),
            'optimizer': np.random.choice(self.search_space['optimizer']),
        }
        return architecture

class ANAS:
    def __init__(self, X, y, search_space=None, num_trials=10, validation_split=0.2):
        self.X = X
        self.y = y
        self.num_trials = num_trials
        self.validation_split = validation_split
        self.search_space = search_space if search_space else ANASearchSpace()
        self.best_architecture = None
        self.best_score = -np.inf

        # Preprocessing: Standardize the data
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        # Split the data into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=self.validation_split, random_state=42
        )

        # Convert data to PyTorch tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32).view(-1, 1)

    def build_model(self, architecture):
        layers = []
        input_dim = self.X_train.shape[1]

        # Add layers according to the architecture
        for _ in range(architecture['num_layers']):
            if architecture['layer_type'] == 'dense':
                layers.append(nn.Linear(input_dim, architecture['num_units']))
                layers.append(architecture['activation']())
                input_dim = architecture['num_units']
            elif architecture['layer_type'] == 'conv':
                layers.append(nn.Conv2d(in_channels=1, out_channels=architecture['num_units'], kernel_size=3, stride=1, padding=1))
                layers.append(architecture['activation']())
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                layers.append(nn.Flatten())
                input_dim = architecture['num_units']  # Update for dense layers after conv

        # Add output layer
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())  # Assuming binary classification

        model = nn.Sequential(*layers)
        return model

    def compile_model(self, model, architecture):
        # Compile the model with the chosen optimizer
        if architecture['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters())
        else:
            optimizer = optim.SGD(model.parameters())
        
        criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
        return criterion, optimizer

    def evaluate_architecture(self, architecture):
        model = self.build_model(architecture)
        criterion, optimizer = self.compile_model(model, architecture)

        # Simple training loop
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(self.X_val)
            val_loss = criterion(val_outputs, self.y_val)
            val_accuracy = ((val_outputs > 0.5) == self.y_val).float().mean().item()
        return val_accuracy  # Return validation accuracy

    def search(self):
        for i in range(self.num_trials):
            architecture = self.search_space.sample_architecture()
            score = self.evaluate_architecture(architecture)
            if score > self.best_score:
                self.best_score = score
                self.best_architecture = architecture
            print(f"Trial {i+1}/{self.num_trials} | Score: {score:.4f} | Architecture: {architecture}")

        print(f"Best Architecture: {self.best_architecture} | Best Score: {self.best_score:.4f}")
        return self.best_architecture
