# Training Code 
import os
import torch
import torchvision
from torchvision.io import read_image
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm  # Updated import for command line usage

# Data Loading
class LandscapeDataset(Dataset):
    def __init__(self, transform=None):
        self.dataroot = './landscape_Images'
        self.images = os.listdir(f'{self.dataroot}/color')
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        color_img = read_image(f'{self.dataroot}/color/{img_path}') / 255
        gray_img = read_image(f'{self.dataroot}/gray/{img_path}') / 255

        if self.transform:
            color_img = self.transform(color_img)
            gray_img = self.transform(gray_img)

        return color_img, gray_img

# Transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
])
dataset = LandscapeDataset(transform=transform)
train_set, test_set = random_split(dataset, [0.8, 0.2])
trainloader = DataLoader(train_set, batch_size=32, shuffle=True)

# Autoencoder Model
class ColorAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1, 64, 3, stride=2)
        self.down2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(128, 3, 3, stride=2, output_padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        d3 = self.relu(self.down3(d2))
        d4 = self.relu(self.down4(d3))
        u1 = self.relu(self.up1(d4))
        u2 = self.relu(self.up2(torch.cat((u1, d3), dim=1)))
        u3 = self.relu(self.up3(torch.cat((u2, d2), dim=1)))
        u4 = self.sigmoid(self.up4(torch.cat((u3, d1), dim=1)))
        return u4

# Training Parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ColorAutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
EPOCHS = 10
for epoch in range(EPOCHS):
    running_loss = 0.0
    for idx, (color_img, gray_img) in tqdm(enumerate(trainloader), total=len(trainloader), desc=f'Epoch {epoch + 1}/{EPOCHS}'):
        color_img = color_img.to(device)
        gray_img = gray_img.to(device)

        optimizer.zero_grad()
        predictions = model(gray_img)
        loss = criterion(color_img, predictions)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch: {epoch + 1}, Loss: {running_loss:.6f}')

# Save the model
torch.save(model.state_dict(), 'color_autoencoder_model.pth')
print("Model saved successfully!")
