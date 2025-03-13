import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define transformations for FashionMNIST (grayscale images)
def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalization for grayscale images
    ])

# Function to load dataset
def get_dataloader(dataset_class, root, train=True, batch_size=64, shuffle=True, transform=None):
    dataset = dataset_class(root=root, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Load FashionMNIST dataset
dataset_class = torchvision.datasets.FashionMNIST  # Changed from CIFAR10 to FashionMNIST
transform = get_transforms()
train_loader = get_dataloader(dataset_class, './data', train=True, transform=transform)
test_loader = get_dataloader(dataset_class, './data', train=False, transform=transform, shuffle=False)

# Define a CNN model for grayscale images
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel for grayscale
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second conv layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))  # Added second conv layer
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))  # Fully connected layer
        x = self.fc2(x)  # Output layer
        return x

# Initialize model, loss function, and optimizer
num_classes = len(train_loader.dataset.classes)
model = CNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
train_losses = []
for epoch in range(epochs):
    epoch_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

# Testing loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')

# Plot training loss
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()

# Function to visualize sample predictions
def visualize_predictions(model, test_loader, class_names):
    model.eval()
    images, labels = next(iter(test_loader))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(5):
        img = images[i].numpy().squeeze()
        img = (img * 0.5) + 0.5  # Denormalize
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Pred: {class_names[predicted[i].item()]}')
        axes[i].axis('off')
    plt.show()

# Class names for FashionMNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Call visualization function
visualize_predictions(model, test_loader, class_names)
