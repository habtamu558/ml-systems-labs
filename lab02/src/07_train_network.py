import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Set random seed
np.random.seed(42)

# Load dataset
digits = load_digits()

X = digits.data / 16.0
y = digits.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Convert labels to one-hot encoding
y_train_encoded = np.eye(10)[y_train]
y_test_encoded = np.eye(10)[y_test]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
print("Training labels:", y_train_encoded.shape)
print("Testing labels:", y_test_encoded.shape)


# Initialize network
width = 16

W1 = np.random.randn(64, width) * 0.01
W2 = np.random.randn(width, width) * 0.01
W3 = np.random.randn(width, 10) * 0.01

b1 = np.zeros(width)
b2 = np.zeros(width)
b3 = np.zeros(10)

print("\nNetwork parameters:")
print("W1:", W1.shape)
print("W2:", W2.shape)
print("W3:", W3.shape)
print("b1:", b1.shape)
print("b2:", b2.shape)
print("b3:", b3.shape)


# ReLU activation function
def relu(Z):
    return np.maximum(0, Z)


# Softmax activation function
def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_values = np.exp(Z)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


# Forward pass
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

    return A3, (Z1, A1, Z2, A2, Z3)


# Cross-entropy loss function
def cross_entropy(predictions, y_true):
    predictions = np.clip(predictions, 1e-12, 1.0)
    loss = -np.sum(y_true * np.log(predictions)) / len(y_true)
    return loss


# Backpropagation
def backward(X, y_true, predictions, cache):
    Z1, A1, Z2, A2, Z3 = cache

    m = X.shape[0]

    # Output layer gradients
    dZ3 = (predictions - y_true) / m

    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0)

    # Second hidden layer gradients
    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (Z2 > 0)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    # First hidden layer gradients
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return dW1, db1, dW2, db2, dW3, db3


# Update network parameters
def update_parameters(gradients, learning_rate):
    global W1, b1, W2, b2, W3, b3

    dW1, db1, dW2, db2, dW3, db3 = gradients

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3


# Test forward pass
X_batch = X_train[:32]
y_batch = y_train_encoded[:32]

predictions, cache = forward(X_batch)

print("\nForward pass:")
print("Predictions shape:", predictions.shape)
print("Probability sum:", np.sum(predictions[0]))

# Test loss
loss = cross_entropy(predictions, y_batch)

print("\nCross-entropy loss:")
print("Loss:", loss)

# Test backpropagation
gradients = backward(X_batch, y_batch, predictions, cache)

print("\nBackpropagation:")
print("W1 gradient:", gradients[0].shape)
print("b1 gradient:", gradients[1].shape)
print("W2 gradient:", gradients[2].shape)
print("b2 gradient:", gradients[3].shape)
print("W3 gradient:", gradients[4].shape)
print("b3 gradient:", gradients[5].shape)

print("\nAll tests completed successfully!")



# Training function
def train(X_train, y_train, epochs, batch_size, learning_rate):

    losses = []

    for epoch in range(epochs):

        # Shuffle the training data
        indices = np.random.permutation(len(X_train))

        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Train using mini-batches
        for start in range(0, len(X_train), batch_size):

            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward pass
            predictions, cache = forward(X_batch)

            # Backpropagation
            gradients = backward(
                X_batch,
                y_batch,
                predictions,
                cache
            )

            # Update weights
            update_parameters(gradients, learning_rate)

        # Calculate loss after each epoch
        predictions, cache = forward(X_train)

        epoch_loss = cross_entropy(predictions, y_train)

        losses.append(epoch_loss)

        print(
            "Epoch", epoch + 1,
            "of", epochs,
            "| Loss:", round(epoch_loss, 6)
        )

    return losses

# Initialize network for a selected width
def initialize_network(width):

    global W1, b1, W2, b2, W3, b3

    W1 = np.random.randn(64, width) * 0.01
    W2 = np.random.randn(width, width) * 0.01
    W3 = np.random.randn(width, 10) * 0.01

    b1 = np.zeros(width)
    b2 = np.zeros(width)
    b3 = np.zeros(10)

# Experiment settings
widths = [16, 64]
learning_rates = [0.001, 0.01, 0.1]

epochs = 50
batch_size = 32

# Store experiment results
all_results = {}

# Run experiments
for width in widths:

    for learning_rate in learning_rates:

        print("\n================================")
        print("Network width:", width)
        print("Learning rate:", learning_rate)
        print("================================")

        # Reset random seed for fair comparison
        np.random.seed(42)

        # Initialize fresh network
        initialize_network(width)

        # Train network
        losses = train(
            X_train,
            y_train_encoded,
            epochs,
            batch_size,
            learning_rate
        )

        # Test the trained network
        predictions, cache = forward(X_test)

        predicted_labels = np.argmax(predictions, axis=1)

        accuracy = np.mean(predicted_labels == y_test)

        # Save results
        all_results[(width, learning_rate)] = {
            "losses": losses,
            "accuracy": accuracy
        }

        print("\nFinal test accuracy:", round(accuracy * 100, 2), "%")

# Plot loss curves
plt.figure(figsize=(10, 6))

for (width, learning_rate), result in all_results.items():

    plt.plot(
        range(1, epochs + 1),
        result["losses"],
        label=f"Width={width}, LR={learning_rate}"
    )

plt.title("Training Loss for Different Network Configurations")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("lab02/results/loss_curves.png")

plt.show()

print("\nLoss curve saved successfully!")

# Measure forward and backward pass time

X_batch = X_train[:32]
y_batch = y_train_encoded[:32]

# Prepare data for timing
predictions, cache = forward(X_batch)

# Warm-up
forward(X_batch)
backward(X_batch, y_batch, predictions, cache)

# Measure forward pass 100 times
forward_times = []

for i in range(100):
    start = time.perf_counter()
    predictions, cache = forward(X_batch)
    end = time.perf_counter()

    forward_times.append(end - start)

# Measure backward pass 100 times
backward_times = []

for i in range(100):
    start = time.perf_counter()
    backward(X_batch, y_batch, predictions, cache)
    end = time.perf_counter()

    backward_times.append(end - start)

# Calculate median times in milliseconds
forward_median = np.median(forward_times) * 1000
backward_median = np.median(backward_times) * 1000

print("\nExecution Time Results:")
print("Forward pass median:", round(forward_median, 4), "ms")
print("Backward pass median:", round(backward_median, 4), "ms")