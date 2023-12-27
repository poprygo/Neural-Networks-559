import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GeometryCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(GeometryCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)  # Output: 100x100
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Output: 50x50 (due to pooling)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 25x25 (due to pooling)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 25 * 25, 128)  
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
class GeometryDataset(Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform
        self.classes = list(set(labels))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(DATA_DIR, self.image_files[idx])
        image = Image.open(img_name)
        label = self.classes.index(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label
    
DATA_DIR = '/Users/yaroslavpopryho/Study/UIC/Classes/Neural Networks/HW5/output'

transform = transforms.Compose([transforms.ToTensor()])

all_files = os.listdir(DATA_DIR)
all_images = [f for f in all_files if f.endswith('.png')]
labels = [f.split('_')[0] for f in all_images]

train_images, test_images = [], []
train_labels, test_labels = [], []

classes = list(set(labels))
for cls in classes:
    class_images = [img for img, label in zip(all_images, labels) if label == cls]
    class_images.sort()
    train_images.extend(class_images[:8000])
    test_images.extend(class_images[8000:])
    train_labels.extend([cls] * 8000)
    test_labels.extend([cls] * 2000)

train_dataset = GeometryDataset(train_images, train_labels, transform=transform)
test_dataset = GeometryDataset(test_images, test_labels, transform=transform)

# Hyperparameters
batch_size = 32
learning_rate = 0.01
epochs = 10


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = GeometryCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in tqdm(range(epochs), desc="Epochs"):
    # Training
    model.train()
    total_train_loss = 0
    total_train_correct = 0

    with tqdm(train_loader, desc="Train Batch", leave=False) as pbar:
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1} Train Loss: {total_train_loss/(batch_idx+1):.4f}")

            _, predicted = torch.max(outputs.data, 1)
            total_train_correct += (predicted == labels).sum().item()

    train_losses.append(total_train_loss / len(train_loader))
    train_accuracies.append(100 * total_train_correct / len(train_dataset))

    # Evaluation
    model.eval()
    total_test_loss = 0
    total_test_correct = 0

    with tqdm(test_loader, desc="Test Batch", leave=False) as pbar:
        for batch_idx, (inputs, labels) in enumerate(pbar):

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_test_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1} Test Loss: {total_test_loss/(batch_idx+1):.4f}")

            _, predicted = torch.max(outputs.data, 1)
            total_test_correct += (predicted == labels).sum().item()

    test_losses.append(total_test_loss / len(test_loader))
    test_accuracies.append(100 * total_test_correct / len(test_dataset))

MODEL_PATH = "0502-650546223-POPRYHO.pth"
torch.save(model.state_dict(), MODEL_PATH)

# Epochs vs Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.title('Epochs vs Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('epochs_vs_loss.png')  

# Epochs vs Accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Testing Accuracy')
plt.title('Epochs vs Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig('epochs_vs_accuracy.png') 