import numpy as np
import librosa

from model import Sequential
from layers import Dense, Relu, Sigmoid

def generate_synthetic_data():
    sr = 16000 
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # 1. Command
    y_command = 0.5 * np.sin(2 * np.pi * 440 * t) 
    mfcc_command = np.mean(librosa.feature.mfcc(y=y_command, sr=sr, n_mfcc=13).T, axis=0)
    
    # 2. Noise
    y_noise = np.random.normal(0, 0.05, len(t))
    mfcc_noise = np.mean(librosa.feature.mfcc(y=y_noise, sr=sr, n_mfcc=13).T, axis=0)
    
    return mfcc_command.reshape(1, -1), mfcc_noise.reshape(1, -1)

x_yes, x_noise = generate_synthetic_data()

all_data = np.concatenate([x_yes, x_noise], axis=0)
mean = np.mean(all_data, axis=0)
std = np.std(all_data, axis=0) + 1e-8

x_yes = (x_yes - mean) / std
x_noise = (x_noise - mean) / std


model = Sequential([
    Dense(13, 32),
    Relu,
    Dense(32, 1, init_type="xavier"),
    Sigmoid,
])

print("Starting smoke test...")
for i in range(101):
    p_yes = model.forward(x_yes)
    model.backward(p_yes - 1.0, lr=0.1)
    
    p_noise = model.forward(x_noise)
    model.backward(p_noise - 0.0, lr=0.1)
    
    if i % 20 == 0:
        print(f"Epoch {i:3}: Command prob={p_yes[0][0]:.4f} | Noise prob={p_noise[0][0]:.4f}")
