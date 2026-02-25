import numpy as np

from model import Sequential, CategoricalCrossEntropy
from layers import *
from data_utils import prepare_data, extract_features, LABEL_MAP

X_train, y_train = prepare_data('dataset/train')
X_val, y_val = prepare_data('dataset/val')

mean = np.mean(X_train)
std = np.std(X_train) + 1e-8
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

linear_model = Sequential([
    Flatten(),
    Dense(416, 64),
    Relu(),
    Dense(64, 3, init_type="xavier"),
    Softmax(),
])

linear_model.fit(
    CategoricalCrossEntropy(),
    X_train, y_train,
    X_val,   y_val,
    epochs=100,
    batch_size=32,
    lr=0.001
)

conv_model = Sequential([
    Conv1D(input_size=13, output_size=16, kernel_size=3, stride=1),
    Relu(),
    Conv1D(input_size=16, output_size=32, kernel_size=3, stride=1),
    Relu(),

    Flatten(),
    Dense(32 * (X_train.shape[-1] - 4), 32), 
    Relu(),
    Dense(32, 3, init_type="xavier"),
    Softmax(),
])
conv_model.fit(
    CategoricalCrossEntropy(),
    X_train, y_train,
    X_val,   y_val,
    epochs=100,
    batch_size=32,
    lr=0.001
)

import os

def run_inference(model, folder_path="inference"):
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    for file_name in files:
        path = os.path.join(folder_path, file_name)
        
        features = extract_features(path)
        
        x = features[np.newaxis, ...] 
        y_pred = model.forward(x)
        
        idx = np.argmax(y_pred, axis=1)[0]
        confidence = np.max(y_pred)
        
        print(f"File: {file_name:20} | Result: {list(LABEL_MAP.keys())[idx]:10} | Conf: {confidence:.4f}")

run_inference(linear_model)
run_inference(conv_model)

