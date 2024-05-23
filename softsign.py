import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Softsign(nn.Module):
    def __init__(self):
        super(Softsign, self).__init__()

    def forward(self, x):
        return x / (1 + torch.abs(x))

def plot_activation_function(activation_fn, title):
    x = torch.linspace(-5, 5, 100)
    y = activation_fn(x)

    plt.figure(figsize=(6, 4))
    plt.plot(x.numpy(), y.numpy())
    plt.title(title)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

def plot_input_output(input_tensor, output_tensor, title):
    plt.figure(figsize=(6, 4))
    plt.plot(input_tensor.numpy(), output_tensor.numpy(), 'ro', markersize=5)
    plt.title(title)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    softsign = Softsign()
    input_tensor = torch.randn(10)
    output_tensor = softsign(input_tensor)

    plot_activation_function(softsign, "Softsign Activation Function")
    plot_input_output(input_tensor, output_tensor, "Input vs. Output (Softsign)")