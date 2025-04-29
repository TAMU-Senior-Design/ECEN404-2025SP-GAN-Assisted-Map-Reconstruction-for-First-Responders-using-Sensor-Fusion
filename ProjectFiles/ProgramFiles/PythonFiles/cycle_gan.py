#NOTES
#kernel: defines size of 2D convoution window which is a small rectangular area that "slides" across the input image to process data
#convolution: "performed by multiplying and accumulating the instantaneous values of the overlapping samples corresponding to two input signals, one of which is flipped"


import torch
import torch.nn as nn

# Define the Generator Network 
# U net style generator: developed for image segmentation
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()  # Initialize the parent class (nn.Module)
        
        # Encoder: Downsampling network to extract features
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # Convolution with 1 input channel, 64 output channels
            # kernel_size=4 (4x4 filter), stride=2 (units shifted by filter), padding=1 (add extra pixel around input border)
            nn.ReLU(inplace=True),  # ReLU activation function (inplace saves memory)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Convolution to increase channels to 128
            nn.BatchNorm2d(128),  # Batch normalization for faster training and stable gradients
            nn.ReLU(inplace=True),  # ReLU activation function
        )
        
        # Decoder: Upsampling network to reconstruct the output
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            # Transposed convolution to increase spatial size, reduce channels to 64
            nn.BatchNorm2d(64),  # Batch normalization
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  
            # Transposed convolution to increase spatial size, reduce channels to 3 (RGB output)
            nn.Tanh(),  # Tanh activation function to scale output to range [-1, 1]
        )

    def forward(self, x):
        # Forward pass through encoder and decoder
        x = self.encoder(x)  # Pass input through the encoder
        x = self.decoder(x)  # Pass encoded features through the decoder
        return x  # Return the output

# Define the Discriminator Network (PatchGAN architecture)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()  # Initialize the parent class (nn.Module)
        
        # PatchGAN-style model: Convolutional layers to classify real vs. fake patches
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),  
            # Convolution with 4 input channels (depth + RGB), 64 output channels
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU activation with negative slope 0.2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  
            # Convolution to increase channels to 128 and further downsample
            nn.BatchNorm2d(128),  # Batch normalization
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU activation
            nn.Conv2d(128, 1, kernel_size=4, padding=1),  
            # Final convolution to produce 1 output channel (real/fake score for each patch)
        )

    def forward(self, x):
        # Forward pass through the PatchGAN discriminator
        return self.model(x)  # Return the output (real/fake patch scores)

