
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

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

# Use a small batch
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

    cache = (Z1, A1, Z2, A2, A3)

    return A3, cache


# Cross-entropy loss
def cross_entropy(y_true, y_pred):

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        np.sum(y_true * np.log(y_pred), axis=1)
    )

    return loss


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

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return dW1, db1


# Calculate analytical gradients
predictions, cache = forward(X_batch)

Z1, A1, Z2, A2, A3 = cache

dW3, db3, dZ3 = backward_output(A2, A3, y_batch)

dW2, db2, dZ2 = backward_hidden2(A1, Z2, dZ3)

dW1, db1 = backward_hidden1(X_batch, Z1, dZ2)


# Numerical gradient checking for W1
epsilon = 1e-4

print("Checking 10 random weights in W1")

passed = 0

for i in range(10):

    row = np.random.randint(0, W1.shape[0])
    col = np.random.randint(0, W1.shape[1])

    original = W1[row, col]

    # Calculate loss with weight slightly increased
    W1[row, col] = original + epsilon
    pred_plus, _ = forward(X_batch)
    loss_plus = cross_entropy(y_batch, pred_plus)

    # Calculate loss with weight slightly decreased
    W1[row, col] = original - epsilon
    pred_minus, _ = forward(X_batch)
    loss_minus = cross_entropy(y_batch, pred_minus)

    # Restore original weight
    W1[row, col] = original

    # Numerical gradient
    numerical = (loss_plus - loss_minus) / (2 * epsilon)

    # Analytical gradient
    analytical = dW1[row, col]

    # Relative error
    error = abs(numerical - analytical) / (
        abs(numerical) + abs(analytical) + 1e-12
    )

    print(
        "Weight:", (row, col),
        "Numerical:", numerical,
        "Analytical:", analytical,
        "Relative error:", error
    )

    if error < 1e-6:
        passed += 1


print("\nPassed:", passed, "out of 10")
print("Required: at least 9 out of 10")
