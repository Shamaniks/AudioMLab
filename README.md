# Voice Command Classifier (from Scratch)

A minimalist Keyword Spotting (KWS) system built entirely in **NumPy**. This project demonstrates the mathematical foundations of neural networks by implementing forward and backward propagation without high-level libraries like PyTorch or TensorFlow.

## 🎯 Project Goal
To build a robust classifier for "Yes", "No", and "Unknown" commands using the **Google Speech Commands Dataset**, focusing on efficient feature extraction and raw matrix operations.

## 🧠 Key Features
- **Zero-Framework Inference**: All neural layers and training logic are written in pure NumPy.
- **Signal Processing**: Audio features are extracted using Mel-frequency cepstral coefficients (MFCC) via `librosa`.
- **Validation Pipeline**: Includes a synthetic "smoke test" to verify model convergence on wave patterns vs. white noise.

## 🛠 Tech Stack
- **Python 3.x**
- **NumPy** (Linear Algebra & Model Logic)
- **Librosa** (Digital Signal Processing)

## 🚀 Getting Started
```bash
# Install dependencies
pip install numpy librosa

# Run the synthetic convergence test
python main.py
```
