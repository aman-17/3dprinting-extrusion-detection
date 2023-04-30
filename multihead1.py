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

dataset = PrintingDefectsDataset(csv_file='./dataset/train.csv',
                                 root_dir='./dataset/images/',
                                 transform=transform)

train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(len(train_loader))

class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualAttentionBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        attention = self.attention(out)
        out = out * attention
        
        out += residual
        out = self.relu1(out)
        
        return out

class MultiHeadResidualAttentionNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MultiHeadResidualAttentionNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.resnet.fc = nn.Identity()

        self.attention1 = ResidualAttentionBlock(in_channels=2048, out_channels=256)
        self.head1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=num_classes)
        )

        self.attention2 = ResidualAttentionBlock(in_channels=2048, out_channels=256)
        self.head2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        x1 = self.attention1(x)
        x1 = x1.view(x1.size(0), -1,1,1)
        x1 = self.head1(x1)

        x2 = self.attention2(x)
        x2 = x2.view(x2.size(0), -1,1,1)
        x2 = self.head2(x2)
        
        return x1, x2

# Instantiate the model and define the loss function and optimizer
model = MultiHeadResidualAttentionNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())



model.to(device)
num_epochs = 10

for epoch in range(num_epochs):
    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0
    
    # Training
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs1, outputs2 = model(images)

        # Compute the loss
        loss1 = criterion(outputs1, labels[:, 0])
        loss2 = criterion(outputs2, labels[:, 1])
        loss = loss1 + loss2

        # Compute gradients and update weights
        loss.backward()
        optimizer.step()

        # Update training loss and accuracy
        train_loss += loss.item() * images.size(0)
        _, preds1 = torch.max(outputs1, 1)
        _, preds2 = torch.max(outputs2, 1)
        train_acc += torch.sum(preds1 == labels[:, 0].data) + torch.sum(preds2 == labels[:, 1].data)

    # Validation
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs1, outputs2 = model(images)

            # Compute the loss
            loss1 = criterion(outputs1, labels[:, 0])
            loss2 = criterion(outputs2, labels[:, 1])
            loss = loss1 + loss2

            # Update validation loss and accuracy
            val_loss += loss.item() * images.size(0)
            _, preds1 = torch.max(outputs1, 1)
            _, preds2 = torch.max(outputs2, 1)
            val_acc += torch.sum(preds1 == labels[:, 0].data) + torch.sum(preds2 == labels[:, 1].data)

    # Compute average loss and accuracy
    train_loss /= len(train_dataset)
    train_acc /= len(train_dataset)
    val_loss /= len(val_dataset)
    val_acc /= len(val_dataset)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
          .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))