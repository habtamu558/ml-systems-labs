import numpy as np
from sklearn.datasets import load_digits

# Load the dataset
digits = load_digits()

# Get images and labels
X = digits.data
y = digits.target

# Scale the pixel values
X = X / 16.0

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# One-hot encode the labels
y_train_encoded = np.eye(10)[y_train]
y_test_encoded = np.eye(10)[y_test]

# Print the shapes
print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

print("Training labels:", y_train_encoded.shape)
print("Testing labels:", y_test_encoded.shape)

# Print information
print("Input shape:", X.shape)
print("Labels shape:", y.shape)

print("First image:")
print(X[0])

print("First label:")
print(y[0])

import matplotlib.pyplot as plt

# Display the first handwritten digit
plt.imshow(X[0].reshape(8, 8), cmap="gray")

plt.title("First digit: " + str(y[0]))
plt.show()