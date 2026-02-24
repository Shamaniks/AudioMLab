import numpy as np

class Layer:
    """Base class for all layers in the framework."""
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size, init_type="he"):
        super().__init__()
        if init_type == "he":
            std = np.sqrt(2.0 / input_size)
        else:
            std = np.sqrt(1.0 / input_size)
            
        self.weights = np.random.randn(input_size, output_size) * std
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        return input_gradient

class Flatten(Layer):
    def forward(self, input_data):
        self.input_shape = input_data.shape
        return input_data.flatten().reshape(1, -1)

    def backward(self, output_gradient, lr):
        return output_gradient.reshape(self.input_shape)

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data):
        self.input = input_data
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_prime(self.input)

class Softmax(Layer):
    def forward(self, input_data):
        shift_x = input_data - np.max(input_data, axis=1, keepdims=True)
        exps = np.exp(shift_x)
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient 

def relu(x): return np.maximum(0, x)
def relu_prime(x): return (x > 0).astype(float)
Relu = Activation(relu, relu_prime)

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_prime(x): 
    s = sigmoid(x)
    return s * (1 - s)
Sigmoid = Activation(sigmoid, sigmoid_prime)
