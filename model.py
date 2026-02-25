import numpy as np

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, output_gradient, lr):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, lr)
        return output_gradient

    def train_epoch(self, n_samples, X_train, y_train, criterion, batch_size, lr):
        epoch_loss = 0
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            y_pred = self.forward(X_batch)
            loss = criterion.forward(y_pred, y_batch)
            epoch_loss += loss * X_batch.shape[0]
            grad = criterion.backward()
            self.backward(grad, lr)
        return epoch_loss / n_samples

    def val_epoch(self, X_val, y_val, criterion):
        y_pred = self.forward(X_val)
        val_loss = criterion.forward(y_pred, y_val)
        return val_loss

    def fit(self, criterion, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, lr=0.01):
        n_samples = X_train.shape[0]
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            train_loss = self.train_epoch(n_samples, X_train, y_train, criterion, batch_size, lr)
            val_loss   = self.val_epoch(X_val, y_val, criterion)
            self.verbose(epoch, epochs, train_loss, val_loss)

    def verbose(self, epoch, epochs, train_loss, val_loss):
        print(f"Epoch {epoch + 1:>{len(str(epochs))}}/{epochs} - train_loss:{train_loss:.4f} val_loss:{val_loss:.4f}")

class CategoricalCrossEntropy:
    def forward(self, y_pred, y_true_idx):
        self.batch_size = y_pred.shape[0]
        self.y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        self.y_true_idx = y_true_idx
        corect_confidences = self.y_pred[np.arange(self.batch_size), y_true_idx]
        return -np.mean(np.log(corect_confidences))

    def backward(self):
        grad = self.y_pred.copy()
        grad[np.arange(self.batch_size), self.y_true_idx] -= 1
        return grad / self.batch_size

