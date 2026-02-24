import numpy as np
import librosa
from model import Model

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

brain = Model(size=13)

print("Starting smoke test...")
for i in range(101):
    p_yes = brain.forward(x_yes)
    brain.backward(1.0, lr=0.1)
    
    p_noise = brain.forward(x_noise)
    brain.backward(0.0, lr=0.1)
    
    if i % 20 == 0:
        print(f"Epoch {i:3}: Command prob={p_yes[0][0]:.4f} | Noise prob={p_noise[0][0]:.4f}")
