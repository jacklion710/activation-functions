# Activation Functions

Study on the topic of activation functions in AI. Focusing on practical examples with code implementations and plots to clearly demonstrate the effect of these transformations on data in an interactive fashion.

I am paraphrasing the concepts from this [Machine Learning Mastery article](https://machinelearningmastery.com/using-activation-functions-in-neural-networks/#:~:text=Activation%20functions%20play%20an%20integral,a%20simple%20linear%20regression%20model.) (with a few small corrections added).

For this repository I have chosen to use Pytorch but the article provides examples in Tensorflow.

* Why are nonlinearities important in a neural network?
* How do different activation functions contribute to the vanishing gradient problem?
* What is the difference between sigmoid, tanh, and ReLU activation functions?
* How do you use different activations in a TensorFlow model?

## Why Do We Need Nonlinear Activation Functions?

Using multiple linear layers is basically the same as using a single layer. In other words you can reqwrite the output layer as a linear combination of the original input variables if you used a linear hidden layer. Basically you could represent the network as a single layer under these circumstances.

![Single hidden layer neural network with linear layers](images/Single%20hidden%20layer%20neural%20network%20with%20linear%20layers.png)

To understand this, let's consider a simple example. Suppose you have a neural network with one hidden layer and linear activation functions. The output of the hidden layer can be written as:

$h = W_1x + b_1$

where $W_1$ is the weight matrix, $x$ is the input vector, and $b_1$ is the bias vector for the hidden layer.

The output of the network can then be written as:

$y = W_2h + b_2$

where $W_2$ is the weight matrix, $h$ is the output of the hidden layer, and $b_2$ is the bias vector for the output layer.

Substituting the expression for $h$ into the output equation, we get:

$y = W_2(W_1x + b_1) + b_2$ $y = W_2W_1x + W_2b_1 + b_2$

This can be rewritten as:

$y = Wx + b$

where $W = W_2W_1$ and $b = W_2b_1 + b_2$.

As you can see, the network with a linear hidden layer can be simplified to a single layer with weights $W$ and bias $b$. This demonstrates that using multiple linear layers does not increase the complexity or representational power of the network.

For more complex functions and to learn non-linear relationships, you need to use non-linear activation functions in the hidden layers of the neural network.

### Sigmoid

$\sigma(x) = \frac{1}{1 + e^{-x}}$

Popular choice because its range (0-1) mimics probability values which is useful for probabilistic outputs.

![Sigmoid activation function](images/Sigmoid%20activation%20function.png)


The derivation of an activation function can be useful to analyze as well due to back propagation and the chain rule which affects the training process.

![Sigmoid activation function (blue) and gradient (orange)](images/Sigmoid%20activation%20function%20(blue)%20and%20gradient%20(orange).png)

The gradient <i>f'(x)</x> is always between 0-0.25. As the x tends towards positive or negative infinity, the gradient tends to 0. This has the potential to cause the vanishing gradient problem meaning the gradient becomes too small to initiate the correction.

Because backpropagation follows the chain rule which states that the gradient of the loss function at each layer is the gradient at its subsequent layer multiplied by the gradient of its activation function, then if the gradient of the activation functions are less than 1, the gradient at some layer far away from the output will be close to 0.

Since sigmoid functions are always less than 1, a network with more layers would exacerbate the vanishing gradient problem. There is a saturation region where the gradient tends to - which is where the magnitude of x is large. So, if the output of the weighted sum of activations from previous layers is large, then you would have a very small gradient propagating through this neuron as the derivative of the activation $\sigma$ with respect to the input to the activation function would be small (in the saturation region).

### Hyperbolic Tangent

The hyperbolic tangent (tanh) activation function is defined as: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

Similar to sigmoid, it has a larger range of output values compared to the sigmoid function and a larger maximum gradient. Also known as tanh, it is a hyperbolic analog to the normal tangent function for circles.

![Tanh activation function](images/Tanh%20activation%20function.png)

Its gradient:

![Tanh activation function (blue) and gradient (orange)](images/Tanh%20activation%20function%20(blue)%20and%20gradient%20(orange).png)

The gradient has a max value of 1, compared to sigmoid where the largest gradient is 0.25. This relieves the vanishing gradient problem in instances where tanh is used. Tanh also has a saturation region where the value of the gradient tends toward 0 as the magnitude of the input x gets larger.

Some key properties of the tanh function:

1. Output range: The output of the tanh function is bounded between -1 and 1, which is symmetric around zero. This can be advantageous in certain scenarios, such as in hidden layers of a neural network, as it allows for both positive and negative activations.
2. Zero-centered: Unlike the sigmoid function, the tanh function is zero-centered, meaning that its output is close to zero for inputs near zero. This property can help speed up the convergence of the neural network during training.
3. Relation to sigmoid: The tanh function is related to the sigmoid function by the following equation: $\tanh(x) = 2\sigma(2x) - 1$ where $\sigma(x)$ is the sigmoid function. This relationship shows that the tanh function can be seen as a scaled and shifted version of the sigmoid function.
4. Vanishing gradient problem: Although the tanh function has a larger maximum gradient compared to the sigmoid function, it still suffers from the vanishing gradient problem for inputs with large magnitudes. The saturation regions of the tanh function, where the gradient approaches zero, can slow down the learning process in deep neural networks.

Despite the advantages of the tanh function over the sigmoid function, it is still susceptible to the vanishing gradient problem. This has led to the development of other activation functions, such as the Rectified Linear Unit (ReLU) and its variants, which have become more popular in modern deep learning architectures.

### Rectified Linear Unit (ReLU)

$x=y$

ReLU is a simple max (0,x) function which can also be thought of as a piecewise function with all inputs less than 0 mapping to 0 and all inputs greater than or equal to 0 mapping back to themselves (identity function).

![ReLU activation function](images/ReLU%20activation%20function.png)

The Gradient:

![ReLU activation function (blue line) and gradient (orange)](images/ReLU%20activation%20function%20(blue%20line)%20and%20gradient%20(orange).png)

Notice how the gradient of ReLU is 1 whenever the input is positive, which helps address the vanishing gradient problem. However, whenever the input is negative, the gradient is 0. This causes another problem, the dead neuron/dying ReLU problem which is an issue if a neuron is persistently inactivated.

In this case, the neuron can no longer learn and its weights are never updated due to the chain rule since it has a 0 gradient as one of its terms. If this happens for all data in your dataset, then it can be very difficult for this neuron to learn from you dataset unless the activations in the previous layer change such that the neuron is no longer "dead".

To address the dead neuron problem, several variants of the ReLU activation function have been proposed:

1. Leaky ReLU: Instead of having a gradient of 0 for negative inputs, Leaky ReLU introduces a small negative slope (typically 0.01) for negative inputs. This allows the neuron to have a non-zero gradient even when the input is negative, helping to alleviate the dead neuron problem. The gradient of Leaky ReLU is defined as:
    - gradient = 1, when input > 0
    - gradient = 0.01 (or another small constant), when input ≤ 0
2. Parametric ReLU (PReLU): Similar to Leaky ReLU, PReLU also has a negative slope for negative inputs. However, instead of using a fixed value, the negative slope is treated as a learnable parameter that can be optimized during training. This allows the network to adapt the negative slope based on the specific problem at hand.
3. Exponential Linear Unit (ELU): ELU is another variant that aims to address the dead neuron problem. For positive inputs, it behaves like the standard ReLU. For negative inputs, it has a negative saturation value that approaches a constant as the input becomes more negative. The ELU function is defined as:
    - $$ f(x) = \begin{cases} x, & \text{when } x > 0\ \alpha(\exp(x) - 1), & \text{when } x \leq 0 \end{cases} $$
    - where $\alpha$ is a hyperparameter that controls the saturation value for negative inputs.

These variants of the ReLU activation function help mitigate the dead neuron problem by allowing gradients to flow even when the input is negative. By using these alternatives, the network can continue to learn and update the weights of neurons that would otherwise become stuck in an inactive state

## Other notable examples

1. Hard Sigmoid:

    * The hard sigmoid function is an approximation of the sigmoid function that is computationally more efficient.
    * It is defined as:
    `hard_sigmoid(x) = max(0, min(1, (x + 1) / 2))`

    * The hard sigmoid function is a piecewise linear approximation of the sigmoid function, which makes it faster to compute.
    * It is often used in quantized neural networks or in scenarios where computational efficiency is a priority.

2. Swish:

    * The Swish activation function is a smooth, non-monotonic function that combines the properties of sigmoid and ReLU.
    * It is defined as:
    `swish(x) = x * sigmoid(x)`

    * Swish has been shown to perform well in deep learning models, especially in combination with batch normalization.
    * It has a smooth gradient and does not suffer from the vanishing gradient problem like the sigmoid function.

3. Hard Tanh:

    * The hard tanh function is a piecewise linear approximation of the tanh function.
    * It is defined as:
    `hard_tanh(x) = max(-1, min(1, x))`

    * Similar to the hard sigmoid, the hard tanh function is computationally more efficient than the standard tanh function.
    * It is often used in quantized neural networks or in scenarios where computational efficiency is a priority.

4. Softsign:

    * The softsign activation function is similar to tanh but has a gentler slope and a larger range.
    * It is defined as:
    `softsign(x) = x / (1 + |x|)`

    * The softsign function is differentiable everywhere and has a smooth gradient.
    * It can be seen as a alternative to tanh, particularly in scenarios where a larger output range is desired.