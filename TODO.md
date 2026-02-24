# Project Roadmap

### Phase 1: Custom Engine (The Framework)
- [x] Initial NumPy model core.
- [x] **Modular Layer API**: Implement base `Layer` class for plug-and-play architecture.
- [ ] **Advanced Layers**:
    - [ ] `Dense`: Fully connected layer with Xavier/He initialization.
    - [ ] **`Conv1D`**: Manual 1D convolution and backprop for temporal audio patterns.
    - [ ] `Flatten` & `Activation` (ReLU/Softmax) layers.
- [ ] **Loss Functions**: Categorical Cross-Entropy for multi-class support.

### Phase 2: Signal Processing & Data
- [ ] **Audio Pipeline**: 
    - [ ] MFCC feature extraction with `librosa`.
    - [ ] Signal normalization and fixed-length padding (1.0s).
- [ ] **Google Speech Commands**: 
    - [ ] Subset integration: `Yes`, `No`, and `Unknown` (distractors).
    - [ ] Efficient data generator for training.

### Phase 3: Training & Delivery
- [ ] **Integration**: Build a `Sequential` model (Conv1D -> ReLU -> Dense -> Softmax).
- [ ] **Validation**: Achieve stable convergence on the 3-class problem.
- [ ] **Demo**: Script for local inference on raw `.wav` files.
- [ ] **Final README**: Documenting the math and performance.
