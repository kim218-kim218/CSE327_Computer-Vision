import numpy as np
from tensorflow.keras.datasets import mnist

# Helper functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(preds, labels):
    return -np.sum(labels * np.log(preds + 1e-8)) / preds.shape[0]

def accuracy(preds, labels):
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(labels, axis=1)
    return np.mean(pred_labels == true_labels)

# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test  = x_test.reshape(-1, 784) / 255.0
y_train = np.eye(10)[y_train]
y_test  = np.eye(10)[y_test]

# Parameters
N_train = x_train.shape[0]
batch_size = 128
epochs = 10
lr = 0.1
Din, H, Dout = 784, 128, 10

# Initialize weights
w1 = np.random.randn(Din, H) * 0.01
b1 = np.zeros((1, H))
w2 = np.random.randn(H, Dout) * 0.01
b2 = np.zeros((1, Dout))

# Training loop
for epoch in range(epochs):
    # Mini-batch SGD
    indices = np.arange(N_train)
    np.random.shuffle(indices)

    for i in range(0, N_train, batch_size):
        batch_idx = indices[i:i + batch_size]
        x = x_train[batch_idx]
        y = y_train[batch_idx]

        # Forward
        z1 = x.dot(w1) + b1
        h = sigmoid(z1)
        z2 = h.dot(w2) + b2
        y_pred = softmax(z2)

        # Loss
        loss = cross_entropy(y_pred, y)

        # Backward
        dz2 = (y_pred - y) / batch_size
        dw2 = h.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dh = dz2.dot(w2.T) * h * (1 - h)
        dw1 = x.T.dot(dh)
        db1 = np.sum(dh, axis=0, keepdims=True)

        # Update weights
        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2

    # Evaluate on test set
    z1_test = x_test.dot(w1) + b1
    h_test = sigmoid(z1_test)
    z2_test = h_test.dot(w2) + b2
    y_pred_test = softmax(z2_test)
    acc = accuracy(y_pred_test, y_test)

    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Accuracy: {acc*100:.2f}%")