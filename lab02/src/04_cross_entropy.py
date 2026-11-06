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

# One-hot encoding
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


# ReLU
def relu(x):
    return np.maximum(0, x)


# Softmax
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

    return A3


# Cross-entropy loss
def cross_entropy(y_true, y_pred):

    epsilon = 1e-15

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    return loss


# Make predictions
predictions = forward(X_batch)

# Calculate loss
loss = cross_entropy(y_batch, predictions)

print("Predictions shape:", predictions.shape)
print("Loss:", loss)
print("Expected loss:", np.log(10))