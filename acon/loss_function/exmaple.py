import torch
import torch.nn as nn
import torch.optim as optim
from AdaptiveLossFunction import AdaptiveLossFunction  # Importing the AdaptiveLossFunction

# Simple feedforward neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        print("Initializing the neural network...")
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Step 3: Set up the model, optimizer, and loss function
input_size = 1
hidden_size = 10
output_size = 1
learning_rate = 0.01
num_epochs = 100

print("Initializing the model, optimizer, and loss function...")
model = SimpleNet(input_size, hidden_size, output_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
adaptive_loss_fn = AdaptiveLossFunction(mode='mse', patience=10, threshold=0.01)

# Step 4: Create a simple dataset
print("Creating the dataset...")
x_train = torch.linspace(-1, 1, 100).reshape(-1, 1)
y_train = 2 * x_train + torch.randn(x_train.size()) * 0.2  # y = 2x + noise

# Step 5: Training loop
print("Starting the training loop...")
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(x_train)
    loss = adaptive_loss_fn.compute_loss(y_train, outputs)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Adapt the loss function mode
    adaptive_loss_fn.adapt_loss_mode(epoch, loss)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Current Mode: {adaptive_loss_fn.mode}')

# Step 6: Evaluate model performance and mode switching
print(f'Total Mode Switches: {adaptive_loss_fn.get_mode_switch_count()}')
print("Training complete.")
