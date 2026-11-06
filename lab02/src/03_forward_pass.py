import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Make results reproducible
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

# Use 32 images as one batch
X_batch = X_train[:32]

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


# ReLU activation function
def relu(x):
    return np.maximum(0, x)


# Softmax activation function
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Forward propagation
def forward(X):

    # First hidden layer
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # Second hidden layer
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    # Output layer
    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    return A3


# Make predictions
output = forward(X_batch)

# Display results
print("Output shape:", output.shape)

print("First image probabilities:")
print(output[0])

print("Sum of probabilities:")
print(np.sum(output[0]))