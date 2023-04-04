import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd

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
        label = self.printing_defects.iloc[idx, 3]  # change this line
        
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(len(train_loader))


class MultiHeadResidualAttentionNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiHeadResidualAttentionNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.resnet.fc = nn.Identity()

        self.head1 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=num_classes)
        )

        self.head2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        x1 = self.head1(x)
        x2 = self.head2(x)
        return x1, x2

model = MultiHeadResidualAttentionNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss1 = criterion(outputs[0], labels)
        loss2 = criterion(outputs[1], labels)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss1 = criterion(outputs[0], labels)
            loss2 = criterion(outputs[1], labels)
            loss = loss1 + loss2

            val_loss += loss.item() * images.size(0)
    
    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

