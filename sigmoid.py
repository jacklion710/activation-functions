import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SigmoidActivation(nn.Module):
    def __init__(self):
        super(SigmoidActivation, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)

def plot_activation_function(activation_fn):
    x = torch.linspace(-10, 10, 100)
    y = activation_fn(x)

    plt.figure(figsize=(6, 4))
    plt.plot(x.numpy(), y.numpy())
    plt.title("Sigmoid Activation Function")
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
    # Create an instance of the sigmoid activation function
    sigmoid_fn = SigmoidActivation()

    # Generate input tensor
    input_tensor = torch.randn(10)
    print("Input Tensor:")
    print(input_tensor)

    # Apply the sigmoid activation function
    output_tensor = sigmoid_fn(input_tensor)
    print("\nOutput Tensor:")
    print(output_tensor)

    # Plot the sigmoid activation function
    plot_activation_function(sigmoid_fn)

    # Plot the input values and their corresponding output values
    plot_input_output(input_tensor, output_tensor)