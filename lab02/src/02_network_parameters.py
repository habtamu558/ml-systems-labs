import numpy as np

# Hidden layer width
W = 16

# Initialize weights
W1 = np.random.randn(64, W) * 0.01
W2 = np.random.randn(W, W) * 0.01
W3 = np.random.randn(W, 10) * 0.01

# Initialize biases
b1 = np.zeros(W)
b2 = np.zeros(W)
b3 = np.zeros(10)

# Print parameter shapes
print("W1:", W1.shape)
print("W2:", W2.shape)
print("W3:", W3.shape)

print("b1:", b1.shape)
print("b2:", b2.shape)
print("b3:", b3.shape)