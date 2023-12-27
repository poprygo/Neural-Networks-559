import os
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

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
    

class GeometryInferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = GeometryCNN().to(device)
model.load_state_dict(torch.load("0502-650546223-POPRYHO.pth", map_location=device))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])

dataset = GeometryInferenceDataset(root_dir="output/", transform=transform)

dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

classes = ["Circle", "Square", "Octagon", "Heptagon", "Nonagon", "Star", "Hexagon", "Pentagon", "Triangle"]

for images, filenames in dataloader:
    images = images.to(device)
    outputs = model(images)
    predictions = outputs.argmax(dim=1)
    for filename, prediction in zip(filenames, predictions):
        print(f"{filename}: {classes[prediction.item()]}")
