import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.where(x > 0, x, self.negative_slope * x)

class ParametricReLU(nn.Module):
    def __init__(self, init_value=0.25):
        super(ParametricReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(init_value))

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * x)

class ELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

def plot_activation_function(activation_fn, title):
    x = torch.linspace(-5, 5, 100)
    with torch.no_grad():
        y = activation_fn(x)

    plt.figure(figsize=(6, 4))
    plt.plot(x.numpy(), y.detach().numpy())
    plt.title(title)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

def plot_input_output(input_tensor, output_tensor, title):
    plt.figure(figsize=(6, 4))
    plt.plot(input_tensor.numpy(), output_tensor.detach().numpy(), 'ro', markersize=5)
    plt.title(title)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Create instances of the modified ReLU activation functions
    leaky_relu = LeakyReLU(negative_slope=0.1)
    parametric_relu = ParametricReLU(init_value=0.25)
    elu = ELU(alpha=1.0)

    # Generate input tensor
    input_tensor = torch.randn(10)
    print("Input Tensor:")
    print(input_tensor)

    # Apply the modified ReLU activation functions
    leaky_relu_output = leaky_relu(input_tensor)
    parametric_relu_output = parametric_relu(input_tensor)
    elu_output = elu(input_tensor)

    print("\nLeaky ReLU Output:")
    print(leaky_relu_output)
    print("\nParametric ReLU Output:")
    print(parametric_relu_output)
    print("\nELU Output:")
    print(elu_output)

    # Plot the modified ReLU activation functions
    plot_activation_function(leaky_relu, "Leaky ReLU Activation Function")
    plot_activation_function(parametric_relu, "Parametric ReLU Activation Function")
    plot_activation_function(elu, "ELU Activation Function")

    # Plot the input values and their corresponding output values for all modified ReLUs
    plot_input_output(input_tensor, leaky_relu_output, "Input vs. Output (Leaky ReLU)")
    plot_input_output(input_tensor, parametric_relu_output, "Input vs. Output (Parametric ReLU)")
    plot_input_output(input_tensor, elu_output, "Input vs. Output (ELU)")