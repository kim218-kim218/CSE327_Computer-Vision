import numpy as np
from tensorflow.keras.datasets import mnist
import time

activation = 'sigmoid' # sigmoid , relu, tanh
loss_type = 'cross_entropy' # cross_entropy or hinge

# Helper functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    return np.maximum(0,z)

def tanh(z):
    return np.tanh(z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(preds, labels):
    return -np.sum(labels * np.log(preds + 1e-8)) / preds.shape[0]

def hinge_loss(scores, labels):
    # scores: raw z2 without softmax
    # labels: one-hot encoded
    correct_scores = np.sum(scores * labels, axis=1, keepdims=True)
    margins = np.maximum(0, scores - correct_scores + 1)
    margins[labels == 1] = 0
    return np.mean(np.sum(margins, axis=1))

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
# ----- 2 hidden layers -----
# Din, H1, H2, Dout = 784, 256, 128, 10
# ---------------------------
weight = 0.01

# Initialize weights
w1 = np.random.randn(Din, H) * weight
b1 = np.zeros((1, H))
w2 = np.random.randn(H, Dout) * weight
b2 = np.zeros((1, Dout))

# ----- 2 hidden layers -----
# w1 = np.random.randn(Din, H1) * weight
# b1 = np.zeros((1, H1))
# w2 = np.random.randn(H1, H2) * weight
# b2 = np.zeros((1, H2))
# w3 = np.random.randn(H2, Dout) * weight
# b3 = np.zeros((1, Dout))
# ---------------------------

print("Activation Function: " , activation)
print("Loss type: ", loss_type)
print("Number of Epochs: " , epochs)
print("Learning Rate: " , lr)
print("Batch Size: " , batch_size)
print("Din, H, Dout: ",Din,H, Dout)
print("weight: ", weight)

# Training loop
start_time = time.time()
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
        if activation == 'sigmoid':
            h = sigmoid(z1)
        elif activation == 'relu':
            h = relu(z1)
        elif activation == 'tanh':
            h = tanh(z1)
        z2 = h.dot(w2) + b2 # raw scores
        if loss_type == 'cross_entropy':
            y_pred = softmax(z2)
        else:
            y_pred = z2

        # Loss
        if loss_type == 'cross_entropy':
            loss = cross_entropy(y_pred, y)
            dz2 = (y_pred - y) / batch_size
        elif loss_type == 'hinge':
            y_pred = z2  # raw scores
            correct_scores = np.sum(z2 * y, axis=1, keepdims=True)
            margins = np.maximum(0, z2 - correct_scores + 1)
            margins[np.arange(z2.shape[0]), np.argmax(y, axis=1)] = 0
            loss = np.mean(np.sum(margins, axis=1))

            dz2 = np.zeros_like(z2)
            dz2[margins > 0] = 1
            dz2[np.arange(z2.shape[0]), np.argmax(y, axis=1)] -= np.sum(margins > 0, axis=1)
            dz2 /= batch_size


        # Backward
        dw2 = h.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        if activation == 'sigmoid':
            dh = dz2.dot(w2.T) * h * (1 - h)
        elif activation == 'relu':
            dh = dz2.dot(w2.T) * (z1 >= 0)
        elif activation == 'tanh':
            dh = dz2.dot(w2.T) * (1 - h ** 2)

        dw1 = x.T.dot(dh)
        db1 = np.sum(dh, axis=0, keepdims=True)

        # Update weights
        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2

    # Evaluate on test set
    z1_test = x_test.dot(w1) + b1
    if activation == 'sigmoid':
        h_test = sigmoid(z1_test)
    elif activation == 'relu':
        h_test = relu(z1_test)
    elif activation == 'tanh':
        h_test = tanh(z1_test)

    z2_test = h_test.dot(w2) + b2
    y_pred_test = softmax(z2_test)
    acc = accuracy(y_pred_test, y_test)

    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Accuracy: {acc*100:.2f}%")

# ----- 2 hidden layers -----------------------
# start_time = time.time()
# for epoch in range(epochs):
#     indices = np.arange(N_train)
#     np.random.shuffle(indices)

#     for i in range(0, N_train, batch_size):
#         batch_idx = indices[i:i + batch_size]
#         x = x_train[batch_idx]
#         y = y_train[batch_idx]

#         # Forward pass
#         z1 = x.dot(w1) + b1
#         h1 = relu(z1) if activation == 'relu' else sigmoid(z1) if activation == 'sigmoid' else tanh(z1)

#         z2 = h1.dot(w2) + b2
#         h2 = relu(z2) if activation == 'relu' else sigmoid(z2) if activation == 'sigmoid' else tanh(z2)

#         z3 = h2.dot(w3) + b3

#         if loss_type == 'cross_entropy':
#             y_pred = softmax(z3)
#             loss = cross_entropy(y_pred, y)
#             dz3 = (y_pred - y) / batch_size
#         else:
#             y_pred = z3
#             correct_scores = np.sum(z3 * y, axis=1, keepdims=True)
#             margins = np.maximum(0, z3 - correct_scores + 1)
#             margins[np.arange(z3.shape[0]), np.argmax(y, axis=1)] = 0
#             loss = np.mean(np.sum(margins, axis=1))

#             dz3 = np.zeros_like(z3)
#             dz3[margins > 0] = 1
#             dz3[np.arange(z3.shape[0]), np.argmax(y, axis=1)] -= np.sum(margins > 0, axis=1)
#             dz3 /= batch_size

#         # Backward pass
#         dw3 = h2.T.dot(dz3)
#         db3 = np.sum(dz3, axis=0, keepdims=True)

#         dh2 = dz3.dot(w3.T)
#         if activation == 'sigmoid':
#             dh2 *= h2 * (1 - h2)
#         elif activation == 'relu':
#             dh2 *= (z2 > 0)
#         elif activation == 'tanh':
#             dh2 *= (1 - h2 ** 2)

#         dw2 = h1.T.dot(dh2)
#         db2 = np.sum(dh2, axis=0, keepdims=True)

#         dh1 = dh2.dot(w2.T)
#         if activation == 'sigmoid':
#             dh1 *= h1 * (1 - h1)
#         elif activation == 'relu':
#             dh1 *= (z1 > 0)
#         elif activation == 'tanh':
#             dh1 *= (1 - h1 ** 2)

#         dw1 = x.T.dot(dh1)
#         db1 = np.sum(dh1, axis=0, keepdims=True)

#         # Update weights
#         w1 -= lr * dw1; b1 -= lr * db1
#         w2 -= lr * dw2; b2 -= lr * db2
#         w3 -= lr * dw3; b3 -= lr * db3

#     # Evaluation
#     z1_test = x_test.dot(w1) + b1
#     h1_test = relu(z1_test) if activation == 'relu' else sigmoid(z1_test) if activation == 'sigmoid' else tanh(z1_test)
#     z2_test = h1_test.dot(w2) + b2
#     h2_test = relu(z2_test) if activation == 'relu' else sigmoid(z2_test) if activation == 'sigmoid' else tanh(z2_test)
#     z3_test = h2_test.dot(w3) + b3

#     y_pred_test = softmax(z3_test) if loss_type == 'cross_entropy' else z3_test
#     acc = accuracy(y_pred_test, y_test)
#     print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Accuracy: {acc*100:.2f}%")
# ------------------------------------------------

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total Training Time: {elapsed_time:.2f} seconds")