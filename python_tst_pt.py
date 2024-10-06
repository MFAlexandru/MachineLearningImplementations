import torch
import time

# Define softmax function using PyTorch
def softmax(x):
    return torch.nn.functional.softmax(x, dim=1)

# Create a random 500x500 tensor on the CPU
tensor = torch.rand(500, 500)

# Time the softmax operation
start_time = time.time()
softmax_result = softmax(tensor)
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
elapsed_time