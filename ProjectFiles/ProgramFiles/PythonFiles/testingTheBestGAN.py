import torch
from torchvision import transforms
from torch import nn
from PIL import Image
import os

# Generator model
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

# Load the pre-trained GAN model
def load_model(model_path, device):
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    generator.eval()
    return generator

# Process the input image
def process_image(image_path, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Open image and apply transform
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor

# Generate output from the model and save it
def save_image(tensor, output_path):
    image = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    image.save(output_path)
    print(f"Output saved to {output_path}")


def main():
    # Input paths
    model_path = "gan_model.pth"  # Path to the trained model
    image_path = "depth_0.png"  # Path to the input depth map image
    output_path = "DemoTest.png"  # Path to save the output image

    # Check if files exist
    if not os.path.exists(model_path):
        print("Error: Model file not found.")
        return
    if not os.path.exists(image_path):
        print("Error: Image file not found.")
        return

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    generator = load_model(model_path, device)

    # Process the input image
    input_tensor = process_image(image_path, device)

    # Generate output
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # Save the result
    save_image(output_tensor, output_path)


if __name__ == "__main__":
    main()
