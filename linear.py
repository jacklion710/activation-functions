import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class LinearActivation(nn.Module):
    def __init__(self):
        super(LinearActivation, self).__init__()

    def forward(self, x):
        return x

def plot_activation_function(activation_fn):
    x = torch.linspace(-5, 5, 100)
    y = activation_fn(x)

    plt.figure(figsize=(6, 4))
    plt.plot(x.numpy(), y.numpy())
    plt.title("Linear Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

def plot_input_output(input_tensor, output_tensor):
    plt.figure(figsize=(6, 4))
    plt.plot(input_tensor.numpy(), output_tensor.numpy(), 'ro', markersize=5)
    plt.title("Input vs. Output")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Create an instance of the linear activation function
    linear_fn = LinearActivation()

    # Generate input tensor
    input_tensor = torch.randn(10)
    print("Input Tensor:")
    print(input_tensor)

    # Apply the linear activation function
    output_tensor = linear_fn(input_tensor)
    print("\nOutput Tensor:")
    print(output_tensor)

    # Plot the linear activation function
    plot_activation_function(linear_fn)

    # Plot the input values and their corresponding output values
    plot_input_output(input_tensor, output_tensor)
