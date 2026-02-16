import numpy as np
import librosa

def generate_synthetic_data():
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # 1. Command
    y_command = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    mfcc_command = np.mean(librosa.feature.mfcc(y=y_command, sr=sr, n_mfcc=13).T, axis=0)
    
    # 2. Noise
    y_noise = np.random.normal(0, 0.1, len(t))
    mfcc_noise = np.mean(librosa.feature.mfcc(y=y_noise, sr=sr, n_mfcc=13).T, axis=0)
    
    return mfcc_command.reshape(1, -1), mfcc_noise.reshape(1, -1)

x_yes, x_noise = generate_synthetic_data()


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

# TEST FITTING
brain = Model()
for i in range(100):
    p_yes = brain.forward(x_yes)
    brain.backward(1.0)
    
    p_noise = brain.forward(x_noise)
    brain.backward(0.0)
    
    if i % 20 == 0:
        print(f"Epoch: {i}: Command prob={p_yes[0][0]:.4f} | Noise prob={p_noise[0][0]:.4f}")
