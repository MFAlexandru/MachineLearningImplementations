import numpy as np
import time

# Define softmax function
def softmax(x):
    # Subtract the max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Create a random 500x500 tensor
tensor = np.random.rand(500, 500)

# Time the softmax operation
start_time = time.time()
softmax_result = softmax(tensor)
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(elapsed_time)