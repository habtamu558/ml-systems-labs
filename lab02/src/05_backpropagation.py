
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Reproducible results
np.random.seed(42)

# Load dataset
digits = load_digits()

X = digits.data / 16.0
y = digits.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# One-hot encode labels
y_train_encoded = np.eye(10)[y_train]

# Use 32 images
X_batch = X_train[:32]
y_batch = y_train_encoded[:32]

# Network width
W = 16

# Initialize weights
W1 = np.random.randn(64, W) * 0.01
W2 = np.random.randn(W, W) * 0.01
W3 = np.random.randn(W, 10) * 0.01

# Initialize biases
b1 = np.zeros(W)
b2 = np.zeros(W)
b3 = np.zeros(10)


# ReLU function
def relu(x):
    return np.maximum(0, x)


# Softmax function
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Forward propagation
def forward(X):

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    # Save values for backpropagation
    cache = (Z1, A1, Z2, A2, A3)

    return A3, cache


# Backpropagation for output layer
def backward_output(A2, A3, y_true):

    m = y_true.shape[0]

    dZ3 = (A3 - y_true) / m

    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0)

    return dW3, db3, dZ3


# Backpropagation for hidden layer 2
def backward_hidden2(A1, Z2, dZ3):

    dA2 = dZ3 @ W3.T

    dZ2 = dA2 * (Z2 > 0)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    return dW2, db2, dZ2

# Backpropagation for hidden layer 1
def backward_hidden1(X, Z1, dZ2):

    # Calculate error going backwards
    dA1 = dZ2 @ W2.T

    # Derivative of ReLU
    dZ1 = dA1 * (Z1 > 0)

    # Calculate gradients
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return dW1, db1

# Run forward propagation
predictions, cache = forward(X_batch)

# Get saved values
Z1, A1, Z2, A2, A3 = cache

# Output-layer gradients
dW3, db3, dZ3 = backward_output(A2, A3, y_batch)

# Hidden layer 2 gradients
dW2, db2, dZ2 = backward_hidden2(A1, Z2, dZ3)

# Hidden layer 1 gradients
dW1, db1 = backward_hidden1(X_batch, Z1, dZ2)

# Display results
print("Output predictions:", predictions.shape)

print("W3 gradient:", dW3.shape)
print("b3 gradient:", db3.shape)

print("W2 gradient:", dW2.shape)
print("b2 gradient:", db2.shape)

print("W1 gradient:", dW1.shape)
print("b1 gradient:", db1.shape)
