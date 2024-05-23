import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class TanhActivation(nn.Module):
    def __init__(self):
        super(TanhActivation, self).__init__()

    def forward(self, x):
        return torch.tanh(x)

def plot_activation_function(activation_fn):
    x = torch.linspace(-5, 5, 100)
    y = activation_fn(x)

    plt.figure(figsize=(6, 4))
    plt.plot(x.numpy(), y.numpy())
    plt.title("Tanh Activation Function")
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
    # Create an instance of the tanh activation function
    tanh_fn = TanhActivation()

    # Generate input tensor
    input_tensor = torch.randn(10)
    print("Input Tensor:")
    print(input_tensor)

    # Apply the tanh activation function
    output_tensor = tanh_fn(input_tensor)
    print("\nOutput Tensor:")
    print(output_tensor)

    # Plot the tanh activation function
    plot_activation_function(tanh_fn)

    # Plot the input values and their corresponding output values
    plot_input_output(input_tensor, output_tensor)