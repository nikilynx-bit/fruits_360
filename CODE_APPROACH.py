# Code-Based AI Project Lifecycle (Python)
'''
This approach follows the standard machine learning lifecycle using Python, **PyTorch**, and **Scikit-Learn**.

## Phase 1: Scoping Project
* **Problem Statement:** Multi-class image classification (131+ classes).
* **Evaluation Metric:** Top-1 Accuracy, Macro F1-Score.
* **Baseline:** Random guess probability is $1/131 \approx 0.7\%$.

## Phase 2: Data Cleaning
The Fruits-360 dataset is highly clean (standardized 100x100 resolution, white background). "Cleaning" in this context refers to **Data Augmentation** to prevent overfitting and normalize pixel values for the CNN.

## Phase 3: Data Analysis (EDA)
Visualizing the distribution of images per class to check for severe class imbalance. We plot the top 10 classes to ensure the training set is representative.

## Phase 4: AI Modeling
We use **Transfer Learning** with a pre-trained **ResNet-18**. We replace the final fully connected layer to match our 131+ output classes and fine-tune the network.

## Phase 5: Evaluation
We evaluate the model on the held-out Test set using a Classification Report (Precision, Recall, F1) and a Confusion Matrix.
'''

## 🐍 Python Implementation

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ==========================================
# 1. SCOPING PROJECT
# ==========================================
# Target: 131 classes
# Architecture: ResNet-18 (Transfer Learning)
# Device: GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. DATA CLEANING & PREPARATION
# ==========================================
# Training transforms include augmentation to prevent overfitting
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(100),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet standards
])

# Test transforms only resize and normalize (no augmentation)
test_transforms = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Paths to the Kaggle dataset folders
train_dir = 'data/fruits-360/Training'
test_dir = 'data/fruits-360/Test'

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

classes = train_data.classes

# ==========================================
# 3. DATA ANALYSIS (EDA)
# ==========================================
print(f"Total Training Images: {len(train_data)}")
print(f"Total Test Images: {len(test_data)}")

# Count images per class
class_counts = {}
for _, label in train_data.samples:
    class_name = train_data.classes[label]
    class_counts[class_name] = class_counts.get(class_name, 0) + 1

# Plot top 10 classes
top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
labels, counts = zip(*top_classes)
sns.barplot(x=list(counts), y=list(labels), palette="viridis")
plt.title("Top 10 Fruit Classes by Image Count")
plt.xlabel("Image Count")
plt.show()

# ==========================================
# 4. AI MODELLING
# ==========================================
# Load Pre-trained ResNet18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Replace final layer for our specific number of classes
model.fc = nn.Linear(num_ftrs, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 5
print(f"\nStarting Training for {epochs} epochs...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

# ==========================================
# 5. EVALUATION
# ==========================================
print("\nEvaluating on Test Set...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

# Plot Confusion Matrix (Limiting to first 10 classes for readability)
subset_labels = np.arange(10)
cm = confusion_matrix(all_labels, all_preds, labels=subset_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes[:10], yticklabels=classes[:10])
plt.title("Confusion Matrix (Subset of Classes)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()
