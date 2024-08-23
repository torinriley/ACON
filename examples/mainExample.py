from acon import ANAS, ContextualAdapter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Initialize the ANAS module to find the best architecture
anas = ANAS(X_train.numpy(), y_train.numpy(), num_trials=10)
best_architecture = anas.search()

# Define the model class using the best architecture found by ANAS
class BestModel(nn.Module):
    def __init__(self, architecture):
        super(BestModel, self).__init__()
        layers = []
        input_dim = X_train.shape[1]
        
        for _ in range(architecture['num_layers']):
            layers.append(nn.Linear(input_dim, architecture['num_units']))
            layers.append(architecture['activation']())
            input_dim = architecture['num_units']

        layers.append(nn.Linear(input_dim, 1))  # Output layer
        layers.append(nn.Sigmoid())  # Sigmoid for binary classification
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Instantiate the model using the best architecture
best_model = BestModel(best_architecture)

# Select the optimizer based on the best architecture
optimizer_choice = best_architecture['optimizer']
if optimizer_choice == 'adam':
    optimizer = optim.Adam(best_model.parameters(), lr=0.001)
else:
    optimizer = optim.SGD(best_model.parameters(), lr=0.001)

# Define the loss function
criterion = nn.BCELoss()

# Training loop with ContextualAdapter
contextual_adapter = ContextualAdapter(best_model, optimizer)

for epoch in range(10):
    best_model.train()
    optimizer.zero_grad()
    outputs = best_model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    # Apply contextual adaptations based on model performance
    contextual_adapter.adapt(epoch, loss.item())

    # Print loss for each epoch
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Evaluate the best model
best_model.eval()
with torch.no_grad():
    accuracy_train = ((best_model(X_train) > 0.5) == y_train).float().mean().item()
    accuracy_test = ((best_model(X_test) > 0.5) == y_test).float().mean().item()

print(f"Final Accuracy using ANAS and Contextual Adaptation: Train: {accuracy_train}, Test: {accuracy_test}")
