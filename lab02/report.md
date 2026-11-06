# Lab 02: Neural Network from Scratch

## 1. Goal

The goal of this lab was to implement and train a two-hidden-layer neural network from scratch using NumPy. The network was designed to classify handwritten digits from the Scikit-learn Digits dataset.

The experiment also aimed to understand forward propagation, cross-entropy loss, backpropagation, gradient checking, training, and execution-time measurement.

## 2. Method

### Dataset

- Dataset: Scikit-learn Digits dataset
- Input features: 64 pixels per image
- Output classes: 10 digits (0–9)
- Input normalization: Divided pixel values by 16
- Training samples: 1,257
- Testing samples: 540
- Train/test split: 70% / 30%
- Random state: 42
- Labels: One-hot encoded

### Neural Network Architecture

The network contained:
- Input layer: 64 neurons
- First hidden layer: 16 or 64 neurons
- Second hidden layer: 16 or 64 neurons
- Output layer: 10 neurons

ReLU activation was used in both hidden layers, and Softmax was used in the output layer.

Weights were initialized using small random values, and biases were initialized to zero.

### Training Configuration

- Epochs: 50
- Batch size: 32
- Learning rates: 0.001, 0.01, 0.1
- Hidden layer widths: 16 and 64
- Optimizer: Manual gradient descent
- Random seed: 42

Six experiments were performed using different combinations of network widths and learning rates.

### Gradient Checking

Numerical gradient checking was performed on 10 randomly selected weights. Nine out of ten checks passed the required relative error threshold of less than 1e-6.

### Execution Time

Forward and backward propagation were measured separately using 100 runs. One warm-up run was performed before measurement, and the median execution time was calculated.

## 3. Results

### Training and Testing Accuracy

| Network Width | Learning Rate | Test Accuracy |
|---:|---:|---:|
| 16 | 0.001 | 10.19% |
| 16 | 0.01 | 10.19% |
| 16 | 0.1 | 67.22% |
| 64 | 0.001 | 10.19% |
| 64 | 0.01 | 10.19% |
| 64 | 0.1 | 94.44% |

The best result was achieved with a network width of 64 and a learning rate of 0.1, reaching 94.44% test accuracy.

### Execution Time

| Operation | Median Execution Time |
|---|---:|
| Forward Pass | 0.1765 ms |
| Backward Pass | 0.3125 ms |

### Loss Curve

The training loss curves for all six experiments were saved in:

`results/loss_curves.png`

The graph compares how different network widths and learning rates affected training loss over 50 epochs.

## 4. Conclusions

1. **Learning rate matters:** A learning rate of 0.1 produced much better results than 0.001 and 0.01 in these experiments. The lower learning rates did not learn effectively within 50 epochs.

2. **Network width affects performance:** The network with 64 neurons in each hidden layer achieved the highest test accuracy of 94.44%, compared with 67.22% for the network with 16 neurons at the same learning rate.

3. **Forward and backward propagation have different costs:** The backward pass took 0.3125 ms, while the forward pass took 0.1765 ms. This shows that calculating gradients requires more computation than making predictions.

Overall, this lab demonstrated how a neural network can be implemented, trained, evaluated, and measured without using a high-level deep learning framework.