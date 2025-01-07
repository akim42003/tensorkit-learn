import tensor_slow

# Create a tensor
t = tensor_slow.Tensor([2, 3])  # A 2x3 tensor

# Set some values
t[[0, 0]] = 1.0
t[[0, 1]] = 2.0
t[[0, 2]] = 3.0
t[[1, 0]] = 4.0
t[[1, 1]] = 5.0
t[[1, 2]] = 6.0

# Print the tensor
print("Original Tensor:")
t.print()

# Transpose the tensor
t_transposed = t.Tp()
print("Transposed Tensor:")
t_transposed.print()
