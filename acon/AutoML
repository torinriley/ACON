import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import numpy as np

class AutoMLPyTorch:
    def __init__(self, models=None, search_space=None, num_trials=10, batch_size=32, epochs=10, validation_split=0.2):
        self.models = models if models else self._default_models()
        self.search_space = search_space if search_space else self._default_search_space()
        self.num_trials = num_trials
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.best_model = None
        self.best_score = -np.inf

    def _default_models(self):
        # Define default models for AutoML
        return {
            'simple_nn': self._build_simple_nn,
            'cnn': self._build_cnn
        }

    def _default_search_space(self):
        # Define default hyperparameter search space
        return {
            'learning_rate': [0.001, 0.01, 0.1],
            'optimizer': ['adam', 'sgd'],
        }

    def _build_simple_nn(self, input_dim):
        # Define a simple neural network model
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def _build_cnn(self, input_dim):
        # Define a simple CNN model
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * (input_dim // 4) * (input_dim // 4), 1),
            nn.Sigmoid()
        )

    def _get_optimizer(self, model, optimizer_name, learning_rate):
        if optimizer_name == 'adam':
            return optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            return optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_split, random_state=42)
        input_dim = X_train.shape[1]
        
        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Convert to PyTorch tensors
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
        
        # Create data loaders
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=self.batch_size)
        
        for trial in range(self.num_trials):
            # Randomly select model and hyperparameters
            model_name = np.random.choice(list(self.models.keys()))
            learning_rate = np.random.choice(self.search_space['learning_rate'])
            optimizer_name = np.random.choice(self.search_space['optimizer'])
            
            # Build and train model
            model = self.models[model_name](input_dim)
            optimizer = self._get_optimizer(model, optimizer_name, learning_rate)
            criterion = nn.BCELoss()

            model.train()
            for epoch in range(self.epochs):
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(X_batch).squeeze()
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()

            # Validate model
            model.eval()
            with torch.no_grad():
                val_loss = 0
                correct = 0
                total = 0
                for X_batch, y_batch in val_loader:
                    y_pred = model(X_batch).squeeze()
                    val_loss += criterion(y_pred, y_batch).item()
                    predicted = (y_pred > 0.5).float()
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
                accuracy = correct / total
            
            # Check if this model is the best so far
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
            
            print(f"Trial {trial + 1}/{self.num_trials}, Model: {model_name}, Accuracy: {accuracy:.4f}")

    def save_best_model(self, file_name):
        if self.best_model is not None:
            dump(self.best_model, file_name)
            print(f"Best model saved as {file_name}.")
        else:
            print("No model was trained.")

# Example Usage:
# Assuming X and y are your input data and labels
# automl = AutoMLPyTorch(num_trials=5, epochs=10)
# automl.fit(X, y)
# automl.save_best_model('best_model.joblib')
