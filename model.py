class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, output_gradient, lr):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, lr)
        return output_gradient
