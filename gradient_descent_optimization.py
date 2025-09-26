import numpy as np
import torch
import matplotlib.pyplot as plt

def simple_gradient_descent():
    """Simple gradient descent for quadratic function"""
    print("=== Simple Gradient Descent ===")

    # Function: f(x) = x^2 + 2x + 1
    # Derivative: f'(x) = 2x + 2
    # Minimum at x = -1

    def f(x):
        return x**2 + 2*x + 1

    def df_dx(x):
        return 2*x + 2

    # Starting point
    x = 5.0
    learning_rate = 0.1
    iterations = 50

    x_history = [x]
    f_history = [f(x)]

    print(f"Starting at x = {x}, f(x) = {f(x):.4f}")

    for i in range(iterations):
        gradient = df_dx(x)
        x = x - learning_rate * gradient

        x_history.append(x)
        f_history.append(f(x))

        if i % 10 == 0:
            print(f"Iteration {i}: x = {x:.4f}, f(x) = {f(x):.4f}, gradient = {gradient:.4f}")

    print(f"Final: x = {x:.4f}, f(x) = {f(x):.4f}")
    print(f"True minimum at x = -1, f(-1) = {f(-1)}")

    return x_history, f_history

def pytorch_gradient_descent():
    """Gradient descent using PyTorch autograd"""
    print("\n=== PyTorch Automatic Gradient Descent ===")

    # Starting point
    x = torch.tensor([5.0], requires_grad=True)
    learning_rate = 0.1
    iterations = 50

    x_history = [x.item()]
    loss_history = []

    print(f"Starting at x = {x.item():.4f}")

    for i in range(iterations):
        # Clear previous gradients
        if x.grad is not None:
            x.grad.zero_()

        # Compute loss
        loss = x**2 + 2*x + 1

        # Backward pass
        loss.backward()

        # Update parameters
        with torch.no_grad():
            x -= learning_rate * x.grad

        x_history.append(x.item())
        loss_history.append(loss.item())

        if i % 10 == 0:
            print(f"Iteration {i}: x = {x.item():.4f}, loss = {loss.item():.4f}, gradient = {x.grad.item():.4f}")

    print(f"Final: x = {x.item():.4f}, loss = {loss.item():.4f}")

    return x_history, loss_history

def linear_regression_gd():
    """Gradient descent for linear regression"""
    print("\n=== Linear Regression with Gradient Descent ===")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + np.random.randn(n_samples, 1)

    # Add bias term
    X_b = np.c_[np.ones((n_samples, 1)), X]

    # Initialize parameters
    theta = np.random.randn(2, 1)
    learning_rate = 0.1
    n_iterations = 1000

    m = len(X_b)
    cost_history = []

    print(f"Initial theta: {theta.flatten()}")

    for iteration in range(n_iterations):
        # Predictions
        predictions = X_b.dot(theta)

        # Cost (MSE)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

        # Gradients
        gradients = (1/m) * X_b.T.dot(predictions - y)

        # Update parameters
        theta = theta - learning_rate * gradients

        if iteration % 200 == 0:
            print(f"Iteration {iteration}: Cost = {cost:.4f}, theta = {theta.flatten()}")

    print(f"Final theta: {theta.flatten()}")
    print(f"True parameters: [4, 3] (bias=4, slope=3)")

    return theta, cost_history, X, y

def stochastic_gradient_descent():
    """Stochastic gradient descent example"""
    print("\n=== Stochastic Gradient Descent ===")

    # Generate data
    np.random.seed(42)
    n_samples = 1000
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + np.random.randn(n_samples, 1)
    X_b = np.c_[np.ones((n_samples, 1)), X]

    # SGD parameters
    n_epochs = 50
    learning_rate = 0.01
    theta = np.random.randn(2, 1)

    print(f"Initial theta: {theta.flatten()}")

    for epoch in range(n_epochs):
        epoch_cost = 0

        for i in range(n_samples):
            # Random sample
            random_index = np.random.randint(n_samples)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            # Prediction and gradient for single sample
            prediction = xi.dot(theta)
            gradient = xi.T.dot(prediction - yi)

            # Update
            theta = theta - learning_rate * gradient

            # Track cost
            epoch_cost += (prediction - yi)**2

        if epoch % 10 == 0:
            avg_cost = epoch_cost[0][0] / n_samples / 2
            print(f"Epoch {epoch}: Avg Cost = {avg_cost:.4f}, theta = {theta.flatten()}")

    print(f"Final SGD theta: {theta.flatten()}")

def adam_optimizer_example():
    """Adam optimizer implementation"""
    print("\n=== Adam Optimizer Example ===")

    # Simple quadratic function optimization
    def f(x):
        return x[0]**2 + x[1]**2 + 0.1*x[0]*x[1]

    def grad_f(x):
        return np.array([2*x[0] + 0.1*x[1], 2*x[1] + 0.1*x[0]])

    # Adam parameters
    x = np.array([5.0, -3.0])
    learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    iterations = 500

    # Adam variables
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    x_history = [x.copy()]

    print(f"Starting at x = {x}")

    for t in range(1, iterations + 1):
        # Compute gradient
        g = grad_f(x)

        # Update biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2

        # Bias correction
        m_corrected = m / (1 - beta1**t)
        v_corrected = v / (1 - beta2**t)

        # Update parameters
        x = x - learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)

        x_history.append(x.copy())

        if t % 100 == 0:
            print(f"Iteration {t}: x = {x}, f(x) = {f(x):.6f}")

    print(f"Final: x = {x}, f(x) = {f(x):.6f}")
    print(f"True minimum at x = [0, 0], f([0, 0]) = 0")

    return x_history

if __name__ == "__main__":
    simple_gradient_descent()
    pytorch_gradient_descent()
    linear_regression_gd()
    stochastic_gradient_descent()
    adam_optimizer_example()