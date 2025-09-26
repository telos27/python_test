import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def generate_sample_data():
    """Generate sample classification data"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (
        torch.FloatTensor(X_train),
        torch.FloatTensor(X_test),
        torch.LongTensor(y_train),
        torch.LongTensor(y_test)
    )

def train_neural_network():
    """Training loop for a simple neural network"""
    print("=== Neural Network Training Loop ===")

    # Generate data
    X_train, X_test, y_train, y_test = generate_sample_data()

    # Model parameters
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 2
    learning_rate = 0.001
    epochs = 100

    # Initialize model, loss, optimizer
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    train_accuracies = []

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        # loss.backward()
        optimizer.step()

        # Calculate accuracy
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_train).float().mean()

        train_losses.append(loss.item())
        train_accuracies.append(accuracy.item())

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_accuracy = (test_predicted == y_test).float().mean()
        print(f'Test Accuracy: {test_accuracy.item():.4f}')

    return model, train_losses, train_accuracies

def batch_training_loop():
    """Training with mini-batches"""
    print("\n=== Mini-batch Training Loop ===")

    # Generate larger dataset
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Model setup
    model = SimpleNN(X_train.shape[1], 64, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    print(f"Training with batches for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        if (epoch + 1) % 10 == 0:
            accuracy = 100 * correct / total
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_accuracy = (test_predicted == y_test).float().mean()
        print(f'Final Test Accuracy: {test_accuracy.item():.4f}')

if __name__ == "__main__":
    model, losses, accuracies = train_neural_network()
    batch_training_loop()
