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

model = Sequential([
    Flatten(),
    Dense(416, 32),
    Relu,
    Dense(32, 3, init_type="xavier"),
    Softmax(),
])

model.fit(
    CategoricalCrossEntropy(),
    X_train, y_train,
    X_val,   y_val,
    epochs=50,
    batch_size=32,
    lr=0.001
)
