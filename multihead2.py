import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class PrintingDefectsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.printing_defects = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.printing_defects)
    
    def __getitem__(self, idx):
        img_name = self.root_dir + self.printing_defects.iloc[idx, 0]
        image = Image.open(img_name)
        label = self.printing_defects.iloc[idx, 3] 
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = PrintingDefectsDataset(csv_file='/kaggle/input/early-detection-of-3d-printing-issues/train.csv',
                                 root_dir='/kaggle/input/early-detection-of-3d-printing-issues/images/',
                                 transform=transform)

train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
print(len(train_loader))

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_acc = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = torch.round(torch.sigmoid(outputs))
        train_acc += torch.sum(preds == labels.data)
    train_loss /= len(train_loader.dataset)
    train_acc = float(train_acc) / len(train_loader.dataset)
    
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs))
            val_acc += torch.sum(preds == labels.data)
    val_loss /= len(val_loader.dataset)
    val_acc = float(val_acc) / len(val_loader.dataset)
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
          .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
