import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Create a simple neural network with multiple layers
class VanishingGradientNet(nn.Module):
    def __init__(self, num_layers):
        super(VanishingGradientNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(1, 10)])
        self.layers.extend([nn.Linear(10, 10) for _ in range(num_layers-2)])
        self.layers.append(nn.Linear(10, 1))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

# Generate some dummy data
x_data = torch.linspace(-10, 10, 100).view(-1, 1)
y_data = torch.sin(x_data)

# Create the models
num_layers = 5
model_vanishing = VanishingGradientNet(num_layers)
model_addressed = VanishingGradientNet(num_layers)

# Replace sigmoid with ReLU in the second model
model_addressed.activation = nn.ReLU()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer_vanishing = optim.SGD(model_vanishing.parameters(), lr=0.01)
optimizer_addressed = optim.SGD(model_addressed.parameters(), lr=0.01)

# Train the models
num_epochs = 1000
print_interval = 100

vanishing_gradients = []
addressed_gradients = []

for epoch in tqdm(range(num_epochs), desc="Training"):
    # Forward pass
    outputs_vanishing = model_vanishing(x_data)
    outputs_addressed = model_addressed(x_data)

    # Compute the loss
    loss_vanishing = criterion(outputs_vanishing, y_data)
    loss_addressed = criterion(outputs_addressed, y_data)

    # Backward pass and optimization
    optimizer_vanishing.zero_grad()
    loss_vanishing.backward()
    optimizer_vanishing.step()

    optimizer_addressed.zero_grad()
    loss_addressed.backward()
    optimizer_addressed.step()

    # Print the loss values at specified intervals
    if (epoch + 1) % print_interval == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Vanishing Loss: {loss_vanishing.item():.4f}, Addressed Loss: {loss_addressed.item():.4f}")

    # Record the gradient magnitudes for each layer
    vanishing_grad_mag = [layer.weight.grad.abs().mean().item() for layer in model_vanishing.layers]
    addressed_grad_mag = [layer.weight.grad.abs().mean().item() for layer in model_addressed.layers]
    vanishing_gradients.append(vanishing_grad_mag)
    addressed_gradients.append(addressed_grad_mag)

# Plot the gradient magnitudes during training
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for layer_idx in range(num_layers):
    plt.plot(range(num_epochs), [grad[layer_idx] for grad in vanishing_gradients], label=f"Layer {layer_idx+1}")
plt.title("Vanishing Gradient - Gradient Magnitudes")
plt.xlabel("Epoch")
plt.ylabel("Gradient Magnitude")
plt.legend()
plt.yscale("log")

plt.subplot(1, 2, 2)
for layer_idx in range(num_layers):
    plt.plot(range(num_epochs), [grad[layer_idx] for grad in addressed_gradients], label=f"Layer {layer_idx+1}")
plt.title("Addressed - Gradient Magnitudes")
plt.xlabel("Epoch")
plt.ylabel("Gradient Magnitude")
plt.legend()
plt.yscale("log")

plt.tight_layout()
plt.show()