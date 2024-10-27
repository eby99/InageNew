import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the ColorAutoEncoder model class
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

# Load the trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ColorAutoEncoder().to(device)
model.load_state_dict(torch.load('color_autoencoder_model.pth', map_location=device))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# Streamlit UI
st.title("Grayscale to Color Image Prediction")
st.write("Upload a grayscale image to see its colorized version.")

# Image Upload
uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    input_image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(input_image, caption="Original Grayscale Image", use_column_width=True)

    # Apply transformations and model prediction
    img_tensor = transform(input_image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(img_tensor)

    # Post-process the output for display
    colorized_image = output.squeeze().cpu().numpy()
    colorized_image = np.moveaxis(colorized_image, 0, -1)  # Rearrange to HWC for display
    colorized_image = np.clip(colorized_image, 0, 1)  # Ensure values are between 0 and 1
    colorized_image = (colorized_image * 255).astype(np.uint8)  # Convert to uint8 for display
    colorized_image = Image.fromarray(colorized_image)

    # Display the colorized image
    st.image(colorized_image, caption="Colorized Image", use_column_width=True)
