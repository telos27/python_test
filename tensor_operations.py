import numpy as np
import torch

def numpy_tensor_operations():
    """Basic tensor operations using NumPy"""
    print("=== NumPy Tensor Operations ===")

    # Create tensors
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8, 9], [10, 11, 12]])

    print(f"Matrix A:\n{a}")
    print(f"Matrix B:\n{b}")

    # Element-wise operations
    print(f"A + B:\n{a + b}")
    print(f"A * B (element-wise):\n{a * b}")

    # Matrix multiplication
    c = np.array([[1, 2], [3, 4], [5, 6]])
    print(f"Matrix C:\n{c}")
    print(f"A @ C (matrix multiplication):\n{a @ c}")

    # Reshaping and transposition
    print(f"A reshaped (3, 2):\n{a.reshape(3, 2)}")
    print(f"A transposed:\n{a.T}")

    # Statistical operations
    print(f"Sum of A: {np.sum(a)}")
    print(f"Mean of A: {np.mean(a)}")
    print(f"Max of A: {np.max(a)}")

def pytorch_tensor_operations():
    """Basic tensor operations using PyTorch"""
    print("\n=== PyTorch Tensor Operations ===")

    # Create tensors
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    print(f"Tensor X:\n{x}")
    print(f"Tensor Y:\n{y}")

    # Element-wise operations
    print(f"X + Y:\n{x + y}")
    print(f"X * Y (element-wise):\n{x * y}")

    # Matrix operations
    z = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    print(f"Tensor Z:\n{z}")
    print(f"X @ Z (matrix multiplication):\n{x @ z}")

    # Gradient computation example
    x_grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y_grad = x_grad ** 2
    loss = y_grad.sum()

    print(f"X with gradient:\n{x_grad}")
    print(f"Y = X^2:\n{y_grad}")
    print(f"Loss = sum(Y): {loss}")

    loss.backward()
    print(f"Gradients (dL/dX):\n{x_grad.grad}")

    # Device operations (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_device = x.to(device)
    print(f"Tensor moved to {device}")

if __name__ == "__main__":
    numpy_tensor_operations()
    pytorch_tensor_operations()