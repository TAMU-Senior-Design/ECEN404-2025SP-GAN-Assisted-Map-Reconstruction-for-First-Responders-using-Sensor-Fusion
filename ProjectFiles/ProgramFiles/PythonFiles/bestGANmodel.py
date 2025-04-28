import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random

# Dataset class
class DepthDataset(Dataset):
    def __init__(self, image_dir, transform=None, mask_prob=0.5):
        self.image_dir = image_dir
        self.transform = transform
        self.mask_prob = mask_prob
        self.images = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        masked_image, mask = self.apply_random_mask(image)
        return masked_image, image, mask

    def apply_random_mask(self, image):
        mask = torch.ones_like(image)
        if random.random() < self.mask_prob:
            # Apply random spots
            num_spots = random.randint(500, 2000)
            for _ in range(num_spots):
                x, y = random.randint(0, image.shape[1] - 1), random.randint(0, image.shape[2] - 1)
                size = random.randint(1, 5)
                mask[:, x:x+size, y:y+size] = 0
        
        # Simulate missing edges
        if random.random() < 0.5:
            edge_width = random.randint(10, 30)
            mask[:, :edge_width, :] = 0
            mask[:, -edge_width:, :] = 0
            mask[:, :, :edge_width] = 0
            mask[:, :, -edge_width:] = 0

        masked_image = image * mask
        return masked_image, mask

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        return self.model(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * (480 // 8) * (640 // 8), 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# Training loop
def train(dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, device, epochs):
    for epoch in range(epochs):
        g_loss_total = 0.0
        d_loss_total = 0.0
        for masked_images, original_images, masks in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            masked_images = masked_images.to(device)
            original_images = original_images.to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            real_outputs = discriminator(original_images)
            fake_images = generator(masked_images)
            fake_outputs = discriminator(fake_images.detach())
            d_loss = criterion(real_outputs, torch.ones_like(real_outputs)) + \
                     criterion(fake_outputs, torch.zeros_like(fake_outputs))
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, torch.ones_like(fake_outputs)) + \
                     criterion(fake_images, original_images)
            g_loss.backward()
            g_optimizer.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Generator Loss: {g_loss_total / len(dataloader):.4f}, "
              f"Discriminator Loss: {d_loss_total / len(dataloader):.4f}")

    return generator

# Main program
def main():
    # Paths
    dataset_path = "depth_images_moms"
    save_model_path = "gan_model.pth"
    
    # Hyperparameters
    batch_size = 4
    lr = 0.0002
    epochs = 20

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transform
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])


    # Dataset and DataLoader
    dataset = DepthDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers and Loss
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Train
    trained_generator = train(dataloader, generator, discriminator, g_optimizer, d_optimizer, criterion, device, epochs)

    # Save model
    torch.save(trained_generator.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")

if __name__ == "__main__":
    main()
