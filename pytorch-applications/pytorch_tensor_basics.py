import torch

print("=== 1. Tensor Creation ===")
# Scalars, Vectors, Matrices
scalar = torch.tensor(5)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])

print(f"Scalar: {scalar}, shape: {scalar.shape}")
print(f"Vector: {vector}, shape: {vector.shape}")
print(f"Matrix: {matrix}, shape: {matrix.shape}\n")

print("=== 2. Random, Zeros, Ones ===")
rand_tensor = torch.rand(2, 3)
zeros_tensor = torch.zeros(2, 2)
ones_tensor = torch.ones(1, 4)

print(f"Random Tensor:\n{rand_tensor}")
print(f"Zeros Tensor:\n{zeros_tensor}")
print(f"Ones Tensor:\n{ones_tensor}\n")

print("=== 3. Tensor Math ===")
a = torch.tensor([2, 3])
b = torch.tensor([4, 5])

print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"Dot product = {torch.dot(a, b)}\n")

print("=== 4. Indexing & Slicing ===")
x = torch.tensor([[10, 20], [30, 40]])
print(f"x = \n{x}")
print(f"x[0] (first row): {x[0]}")
print(f"x[:, 1] (second column): {x[:, 1]}")
print(f"x[1, 0] (row 1, col 0): {x[1, 0]}\n")

print("=== 5. Reshaping Tensors ===")
original = torch.arange(0, 12)
reshaped = original.view(3, 4)
print(f"Original: {original}")
print(f"Reshaped to 3x4:\n{reshaped}\n")

print("=== 6. Gradients & Autograd ===")
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x**2 + 2 * x
y_sum = y.sum()
y_sum.backward()  # Calculates gradients

print(f"x: {x}")
print(f"y = x**2 + 2x: {y}")
print(f"Gradients dy/dx: {x.grad}\n")

print("=== 7. Move to GPU if available ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_gpu = torch.rand(2, 2).to(device)
print(f"Tensor on device ({device}):\n{tensor_gpu}")

