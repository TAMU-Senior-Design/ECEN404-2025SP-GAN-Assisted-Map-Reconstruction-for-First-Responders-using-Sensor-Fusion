import os
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from cycle_gan import Generator, Discriminator

# Hyperparameters
lr = 0.0002  # Learning rate for the Adam optimizer
betas = (0.5, 0.999)  # Beta parameters for Adam optimizer (controls moment decay rates)
num_epochs = 50  # Total number of training epochs
batch_size = 8  # Number of samples per training batch

# Dataset setup
class DepthDataset(Dataset):
    """
    Custom dataset class for loading paired depth and RGB images.

    Parameters:
    - depth_dir (str): Directory containing depth images.
    - rgb_dir (str): Directory containing RGB images.
    - transform (callable, optional): Optional image transformations to apply to both depth and RGB images.
    """
    def __init__(self, depth_dir, rgb_dir, transform=None):
        # Get sorted list of file paths for depth and RGB images
        self.depth_paths = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.png')])
        self.rgb_paths = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.png')])
        self.transform = transform  # Transformations (e.g., resizing, normalization)

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.depth_paths)

    def __getitem__(self, idx):
        # Load depth and RGB images at the specified index
        depth = Image.open(self.depth_paths[idx]).convert("L")  # Convert depth image to grayscale
        rgb = Image.open(self.rgb_paths[idx]).convert("RGB")   # Convert RGB image to 3-channel format
        if self.transform:
            # Apply transformations if specified
            depth = self.transform(depth)
            rgb = self.transform(rgb)
        return depth, rgb  # Return depth and RGB tensors as a pair

# Data transformations and DataLoader
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors (scale to [0, 1])
])

# Paths to datasets
data_dir = "./depth_vi"  # Directory containing depth images
rgb_dir = "./color"  # Directory containing RGB images

# Initialize dataset and DataLoader
dataset = DepthDataset(data_dir, rgb_dir, transform=transform)  # Create custom dataset instance
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Create DataLoader with shuffling

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
G_depth_to_rgb = Generator().to(device)  # Generator for converting depth to RGB
G_noisy_to_clean = Generator().to(device)  # Generator for denoising depth images
D_rgb = Discriminator().to(device)  # Discriminator for real/fake RGB images
D_clean_depth = Discriminator().to(device)  # Discriminator for real/fake clean depth images

# Optimizers and loss functions
optimizer_G = optim.Adam(
    list(G_depth_to_rgb.parameters()) + list(G_noisy_to_clean.parameters()), 
    lr=lr, betas=betas
)  # Adam optimizer for the generators

optimizer_D = optim.Adam(
    list(D_rgb.parameters()) + list(D_clean_depth.parameters()), 
    lr=lr, betas=betas
)  # Adam optimizer for the discriminators

cycle_loss = torch.nn.L1Loss()  # L1 loss for cycle consistency (pixel-wise difference)
adversarial_loss = torch.nn.MSELoss()  # Mean Squared Error loss for adversarial training

# Training loop
for epoch in range(num_epochs):
    for depth, rgb in dataloader:
        depth, rgb = depth.to(device), rgb.to(device)  # Move data to the GPU/CPU

        # Update Generators
        optimizer_G.zero_grad()  # Reset gradients for the generators
        fake_rgb = G_depth_to_rgb(depth)  # Generate fake RGB images from depth inputs
        fake_depth = G_noisy_to_clean(depth)  # Generate denoised depth from noisy inputs

        # Expand depth image to 3 channels for comparison with RGB
        depth_3ch = depth.repeat(1, 3, 1, 1)  # Repeat depth channel to match RGB format

        # Compute generator losses
        loss_G_rgb = cycle_loss(fake_rgb, rgb)  # L1 loss between generated and real RGB
        loss_G_depth = cycle_loss(fake_depth, depth_3ch)  # L1 loss between generated and real depth
        loss_G = loss_G_rgb + loss_G_depth  # Total generator loss (sum of individual losses)
        loss_G.backward()  # Compute gradients via backpropagation
        optimizer_G.step()  # Update generator parameters

        # Update Discriminators
        optimizer_D.zero_grad()  # Reset gradients for the discriminators

        # Prepare real and fake input pairs for the discriminator
        real_input = torch.cat((depth, rgb), dim=1)  # Concatenate depth and real RGB along the channel axis
        fake_input = torch.cat((depth, fake_rgb.detach()), dim=1)  # Concatenate depth and fake RGB (detach to avoid updating generator)

        # Compute discriminator predictions
        real_validity = D_rgb(real_input)  # Discriminator's prediction for real inputs
        fake_validity = D_rgb(fake_input)  # Discriminator's prediction for fake inputs

        # Compute discriminator losses
        loss_D_real = adversarial_loss(real_validity, torch.ones_like(real_validity))  # Loss for real inputs
        loss_D_fake = adversarial_loss(fake_validity, torch.zeros_like(fake_validity))  # Loss for fake inputs
        loss_D = (loss_D_real + loss_D_fake) / 2  # Average discriminator loss
        loss_D.backward()  # Compute gradients via backpropagation
        optimizer_D.step()  # Update discriminator parameters

    # Log progress for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss G: {loss_G.item():.4f} - Loss D: {loss_D.item():.4f}")

# Save trained models to disk
os.makedirs("saved_models_demo", exist_ok=True)  # Create directory if it doesn't exist
torch.save(G_depth_to_rgb.state_dict(), "saved_models_demo/G_depth_to_rgb.pth")  # Save depth-to-RGB generator
torch.save(G_noisy_to_clean.state_dict(), "saved_models_demo/G_noisy_to_clean.pth")  # Save noisy-to-clean generator
