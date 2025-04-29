import torch
from PIL import Image
from torchvision import transforms
from cycle_gan import Generator
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d  # For 3D point cloud creation and visualization

# Check device availability (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained models
# Generator for depth-to-RGB translation
G_depth_to_rgb = Generator().to(device)  # Initialize the depth-to-RGB generator model on the appropriate device
G_depth_to_rgb.load_state_dict(torch.load("saved_models_demo/G_depth_to_rgb.pth", map_location=device))  
# Load pre-trained weights for the depth-to-RGB generator

# Generator for noisy-to-clean depth translation
G_noisy_to_clean = Generator().to(device)  # Initialize the noisy-to-clean generator model on the appropriate device
G_noisy_to_clean.load_state_dict(torch.load("saved_models_demo/G_noisy_to_clean.pth", map_location=device))  
# Load pre-trained weights for the noisy-to-clean generator

# Preprocessing transformations for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize input images to 256x256 to match the model's input requirements
    transforms.ToTensor(),  # Convert images to PyTorch tensors and scale pixel values to [0, 1]
])

def process_images(depth_image_path):
    """
    Processes a depth image to generate an improved depth map and inferred RGB image.

    Parameters: depth_image_path (str): Path to the input depth image.

    Returns:
    - improved_depth (np.ndarray): Noise-removed depth map.
    - generated_rgb (np.ndarray): RGB image inferred from the depth map.
    """
    # Load and preprocess the depth image
    depth_image = Image.open(depth_image_path).convert("L")  # Open image and convert to grayscale
    input_tensor = transform(depth_image).unsqueeze(0).to(device)  
    # Apply transformations, add a batch dimension, and move to the appropriate device (GPU/CPU)

    # Generate improved depth map using the noisy-to-clean generator
    improved_depth = G_noisy_to_clean(input_tensor).detach().cpu().squeeze(0).numpy()  
    # Pass the input through the model, detach gradients, move to CPU, and convert to NumPy array
    if improved_depth.ndim == 3:
        improved_depth = improved_depth.mean(axis=0)  # If multi-channel, take the mean to create a single-channel depth map

    # Generate inferred RGB image using the depth-to-RGB generator
    generated_rgb = G_depth_to_rgb(input_tensor).detach().cpu().squeeze().numpy().transpose(1, 2, 0)  
    # Process the input through the model and rearrange dimensions to match the (H, W, C) format
    generated_rgb = (generated_rgb + 1) / 2.0  # Normalize pixel values from [-1, 1] to [0, 1]

    return improved_depth, generated_rgb  # Return processed depth map and RGB image

def create_point_cloud(depth_map, rgb_image):
    """
    Creates a point cloud from a depth map and corresponding RGB image.

    Parameters:
    - depth_map (np.ndarray): 2D depth map array.
    - rgb_image (np.ndarray): 3D RGB image array corresponding to the depth map.

    Returns: point_cloud (o3d.geometry.PointCloud): Generated 3D point cloud.
    """
    # Depth map dimensions
    h, w = depth_map.shape  # Get height and width of the depth map

    # Intrinsic parameters (approximated for simplicity)
    fx, fy = w / 2.0, h / 2.0  # Focal lengths (assume square pixels)
    cx, cy = w / 2.0, h / 2.0  # Principal point (center of the image)

    points = []  # List to store 3D points
    colors = []  # List to store corresponding RGB colors

    # Iterate over each pixel in the depth map
    for v in range(h):  # Loop through rows (height)
        for u in range(w):  # Loop through columns (width)
            z = depth_map[v, u]  # Depth value for the current pixel
            if z > 0:  # Consider only valid depth values (non-zero)
                # Compute 3D coordinates (x, y, z) from pixel coordinates (u, v) and depth value (z)
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append((x, y, z))  # Append the 3D point
                colors.append(rgb_image[v, u])  # Append the corresponding RGB color

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))  # Assign 3D points to the point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))  # Assign RGB colors to the point cloud

    return point_cloud  # Return the generated point cloud

# Define the path to the input depth image
depth_image_path = "Test_file.png"  # Replace with the path to your depth image

# Process the depth image to get the improved depth map and inferred RGB image
improved_depth, inferred_rgb = process_images(depth_image_path)

# Display the improved depth map and inferred RGB image side-by-side
plt.figure(figsize=(10, 5))  # Create a figure with specified size

# Improved depth map visualization
plt.subplot(1, 2, 1)  # Create subplot for depth map
plt.imshow(improved_depth, cmap='gray')  # Display depth map in grayscale
plt.title("Improved Depth Map")  # Add a title
plt.axis("off")  # Remove axes for better visualization

# Inferred RGB image visualization
plt.subplot(1, 2, 2)  # Create subplot for RGB image
plt.imshow(inferred_rgb)  # Display inferred RGB image
plt.title("Inferred RGB Image")  # Add a title
plt.axis("off")  # Remove axes for better visualizationm 
# Create a 3D point cloud from the improved depth map and inferred RGB image
point_cloud = create_point_cloud(improved_depth, inferred_rgb)

# Visualize the point cloud using Open3D
o3d.visualization.draw_geometries([point_cloud])  # Display the 3D point cloud
