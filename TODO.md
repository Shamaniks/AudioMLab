# Project Roadmap

### Phase 1: Custom Engine (The Framework)
- [x] Initial NumPy model core.
- [x] **Modular Layer API**: Implement base `Layer` class for plug-and-play architecture.
- [x] **Advanced Layers**:
    - [x] `Dense`: Fully connected layer with Xavier/He initialization.
    - [x] **`Conv1D`**: Manual 1D convolution and backprop for temporal audio patterns.
    - [x] `Flatten` & `Activation` (ReLU/Softmax) layers.
- [x] **Loss Functions**: Categorical Cross-Entropy for multi-class support.

### Phase 2: Signal Processing & Data
- [x] **Audio Pipeline**: 
    - [x] MFCC feature extraction with `librosa`.
    - [x] Signal normalization and fixed-length padding (1.0s).
- [ ] **Google Speech Commands**: 
    - [x] Subset integration: `Yes`, `No`, and `Unknown` (distractors).
    - [ ] Efficient data generator for training.

### Phase 3: Training & Delivery
- [x] **Integration**: Build a `Sequential` model (Conv1D -> ReLU -> Dense -> Softmax).
- [x] **Validation**: Achieve stable convergence on the 3-class problem.
- [x] **Demo**: Script for local inference on raw `.wav` files.
- [ ] **Final README**: Documenting the math and performance.

Phase 4: Custom DSP Engine (Signal Processing from Scratch)
- [ ] **Fourier Analysis**: Implement STFT (Short-Time Fourier Transform) using pure math/NumPy.
- [ ] **Mel Filterbank**: Manual construction of triangular filters for frequency warping.
- [ ] **MFCC Core**: Discrete Cosine Transform (DCT) implementation for final feature decorrelation.
- [ ] **Math Parity**: Comparative analysis vs librosa.feature.mfcc (target MSE < 1e-7).

### Phase 5: Engineering & Quality
- [ ] **Math Verification**: Gradient check for manual Backprop in `Conv1D`.
- [ ] **Reproducibility**: 
    - [ ] Weights Checkpointing: Save/Load model state for training resumes.
    - [ ] Fixed seed integration across NumPy/TensorFlow for deterministic results.
- [ ] **Analysis & Visuals**:
    - [ ] Plotting training dynamics (Loss/Accuracy curves).
    - [ ] Feature visualization: Compare input waveform vs extracted MFCC spectrogram.
- [ ] **CI/CD Infrastructure**:
    - [ ] **GitHub Actions**: Automated linting (`flake8`/`black`) on every push.
    - [ ] Basic unit tests for layer output shapes and loss convergence.
