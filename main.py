import numpy as np

from model import Sequential, CategoricalCrossEntropy
from layers import *
from data_utils import prepare_data

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
    epochs=300,
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
    epochs=300,
    batch_size=32,
    lr=0.001
)
