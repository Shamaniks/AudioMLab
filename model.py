import numpy as np

class Model:
    def __init__(self, size=13):
        self.W = np.random.randn(size, 1) * np.sqrt(1 / size)
        self.b = 0.0

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.W) + self.b
        self.a = 1 / (1 + np.exp(-self.z)) # Sigmoid
        return self.a

    def backward(self, y_true, lr=0.1):
        da = self.a - y_true
        
        dW = np.dot(self.x.T, da)
        db = np.sum(da)
        
        self.W -= lr * dW
        self.b -= lr * db
