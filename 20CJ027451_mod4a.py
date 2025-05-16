import numpy as np
# Weighted sums of the five neurons
weighted_sums = np.array([1.2, 0.9, 0.75, 0.8, 1.5])

# i) Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / e_x.sum()

# ii) ReLU
def relu(x):
    return np.maximum(0, x)

# iii) Bipolar Sigmoid with sigma = 1
def bipolar_sigmoid(x, sigma=1):
    return (2 / (1 + np.exp(-sigma * x))) - 1

softmax_output = softmax(weighted_sums)
relu_output = relu(weighted_sums)
bipolar_sigmoid_output = bipolar_sigmoid(weighted_sums)

print("Softmax Output:", softmax_output)
print("ReLU Output:", relu_output)
print("Bipolar Sigmoid Output:", bipolar_sigmoid_output)
