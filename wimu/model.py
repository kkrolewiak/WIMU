import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import Dataset

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(67712 , num_classes) 
        # for 200x200 change to 12800
        # for 300x300 change to 32768
        # for 400x400 change to 67712 
        # for 500x500 change to 107648
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = f.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = f.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = f.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = f.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        #x = torch.sigmoid(x)
        return x

transform = transforms.Compose([
    transforms.CenterCrop((1589-44, 1580-44)),
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

class MultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=transform):
        self.annotations = pd.read_csv(csv_file, sep=",")
        self.annotations = self.annotations.drop(self.annotations.columns[0], axis=1)
        self.image_dir = image_dir
        self.transform = transform

        self.label_columns = self.annotations.columns[1:]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("L")

        labels = self.annotations.iloc[idx, 1:].values.astype(float)
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels
    
    def num_classes(self):
        return self.label_columns.shape[0]