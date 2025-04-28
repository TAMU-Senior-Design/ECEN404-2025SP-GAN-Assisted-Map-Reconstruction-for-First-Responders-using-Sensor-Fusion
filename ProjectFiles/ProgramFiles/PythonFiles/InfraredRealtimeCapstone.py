import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import time
import open3d as o3d
import asyncio
import websockets
import json

# ----------------------------
# Generator Model Definition
# ----------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# ----------------------------
# Load or Initialize Models
# ----------------------------
def load_or_initialize_models(save_path='models', version=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    generator_path = os.path.join(save_path, f"{version}generator.pth")
    if os.path.exists(generator_path):
        generator.load_state_dict(torch.load(generator_path, map_location=device, weights_only=True))
        print("Generator loaded successfully.")
    else:
        print("No saved generator model found, initializing new model.")
    return generator

# ----------------------------
# Image Generation using GAN
# ----------------------------
def generate_image(generator, input_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    generator.eval()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    try:
        input_image_pil = Image.fromarray(input_image)
    except Exception as e:
        print("Error converting input image to PIL:", e)
        return None
    input_tensor = transform(input_image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        generated_image = generator(input_tensor)

    generated_image = generated_image.squeeze(0).cpu()
    generated_pil = transforms.ToPILImage()(generated_image)
    return generated_pil

# ----------------------------
# Point Cloud Creation with Downsampling and Debugging
# ----------------------------
def display_image(thermal_img, depth_img, display=False):
    scale = 0.01

    print("Starting point cloud creation.")
    img_1 = thermal_img
    img_2 = depth_img

    print("Original depth image shape:", img_2.shape)
    print("Original thermal image shape:", img_1.shape)
    
    sampling_rate = 4
    h, w = img_2.shape
    print("Downsampling with sampling rate:", sampling_rate)
    try:
        grid_y, grid_x = np.meshgrid(np.arange(0, h, sampling_rate),
                                     np.arange(0, w, sampling_rate), indexing='ij')
        print("Meshgrid created. Shapes:", grid_y.shape, grid_x.shape)
    except Exception as e:
        print("Error during meshgrid creation:", e)
        return None

    try:
        depth_sampled = img_2[::sampling_rate, ::sampling_rate]
        pts = np.stack((
            grid_y.flatten(),
            grid_x.flatten(),
            -depth_sampled.astype(np.float32).flatten()
        ), axis=1)
        pts = pts * scale
        pts = np.ascontiguousarray(pts)
        print("Points array created with shape:", pts.shape)
    except Exception as e:
        print("Error stacking points:", e)
        return None

    expected_shape = (depth_sampled.shape[0], depth_sampled.shape[1])
    if img_1.shape[0] != expected_shape[0] or img_1.shape[1] != expected_shape[1]:
        try:
            thermal_resized = cv2.resize(img_1, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_CUBIC)
            print("Resized thermal image to match downsampled depth image shape:", thermal_resized.shape)
        except Exception as e:
            print("Error resizing thermal image:", e)
            return None
    else:
        thermal_resized = img_1

    if len(thermal_resized.shape) == 2:
        thermal_resized = cv2.merge([thermal_resized, thermal_resized, thermal_resized])
    try:
        cols = (thermal_resized.astype(np.float32) / 255.0).reshape(-1, 3)
        cols = np.ascontiguousarray(cols)
        print("Color array created with shape:", cols.shape)
    except Exception as e:
        print("Error processing colors:", e)
        return None

    # Debug: Check data types before Open3D assignment
    print("pts dtype:", pts.dtype, "shape:", pts.shape)
    print("cols dtype:", cols.dtype, "shape:", cols.shape)

    # Convert to float64 as a test (you can remove this if not needed)
    pts = pts.astype(np.float64)
    cols = cols.astype(np.float64)

    # Create Open3D point cloud
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        print("Point cloud object created successfully.")
    except Exception as e:
        print("Error creating point cloud:", e)
        return None

    print("Point cloud created with", pts.shape[0], "points.")
    if display:
        o3d.visualization.draw_geometries([pcd])
    return pcd

# ----------------------------
# Thermal Frame Capture
# ----------------------------
def thermal_frame():
    print("Capturing thermal frame from camera...")
    vc = cv2.VideoCapture(3)
    time.sleep(1)
    rval, frame = vc.read()

    frame = cv2.flip(frame,0)

    #pil_image = Image.fromarray(frame)
    #pil_image.show(title=f"Function {counter}")
    if not rval:
        print("Error: Unable to capture frame from camera.")
        vc.release()
        return None
    try:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print("Error converting frame to grayscale:", e)
        vc.release()
        return None
    target_width = 640
    target_height = 480
    try:
        frame_resized = cv2.resize(frame_gray, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print("Error resizing thermal frame:", e)
        vc.release()
        return None
    print("Thermal frame captured and resized to:", frame_resized.shape)
    #pil_image = Image.fromarray(frame_resized)
    #pil_image.show(title=f"Function {counter}")
    vc.release()
    return frame_resized

# Load the GAN models once at startup
thermal_generator = load_or_initialize_models()
depth_generator = load_or_initialize_models(version="depth_")

# ----------------------------
# WebSocket Streaming to Unity
# ----------------------------
async def stream_to_unity(websocket):
    print("Unity connected. Starting stream...")
    while True:
        print("############## Start Loop ##############")
        # Capture a thermal frame and create a corresponding depth image
        t_img = thermal_frame()
        if t_img is None:
            print("Thermal frame not captured, skipping iteration.")
            await asyncio.sleep(1.0)
            continue
        print("Thermal frame shape:", t_img.shape)
        d_img = cv2.bitwise_not(t_img)
        print("Depth image (inverted thermal) created.")

        #pil_image = Image.fromarray(t_img)
        #pil_image.show(title=f"Not function {counter}")

        # Generate images using the GAN models
        print("Generating GAN image for thermal frame...")
        ig_1 = generate_image(thermal_generator, t_img)
        if ig_1 is None:
            print("Failed to generate thermal GAN image, skipping iteration.")
            await asyncio.sleep(1.0)
            continue
        ig_1 = np.array(ig_1)
        print("Thermal GAN image shape:", ig_1.shape)
        ig_1 = cv2.merge([ig_1, ig_1, ig_1])  # Ensure 3 channels for color

        print("Generating GAN image for depth frame...")
        ig_2 = generate_image(depth_generator, d_img)
        if ig_2 is None:
            print("Failed to generate depth GAN image, skipping iteration.")
            await asyncio.sleep(1.0)
            continue
        ig_2 = np.array(ig_2)
        print("Depth GAN image shape:", ig_2.shape)

        # Create a point cloud (without displaying it during streaming)
        pcd = display_image(ig_1, ig_2, display=False)
        if pcd is None:
            print("Failed to create point cloud, skipping iteration.")
            await asyncio.sleep(1.0)
            continue

        # Convert point cloud to JSON using vectorized operation
        try:
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            print("Converting point cloud to list for JSON serialization...")
            point_cloud = np.hstack((points, colors)).tolist()
            print("Point cloud converted. Total points:", len(point_cloud))
        except Exception as e:
            print("Error converting point cloud:", e)
            await asyncio.sleep(1.0)
            continue
        
        try:
            msg = json.dumps(point_cloud)
            print(f"Sending point cloud with {len(point_cloud)} points to Unity. JSON length: {len(msg)}")
            await websocket.send(msg)
        except Exception as e:
            print("Error sending point cloud:", e)
        await asyncio.sleep(5.0)
        #counter += 1
        print("############## End Loop ##############\n")

async def main():
    WEBSOCKET_PORT = 4321
    print(f"Starting WebSocket server on ws://localhost:{WEBSOCKET_PORT}")
    server = await websockets.serve(stream_to_unity, "localhost", WEBSOCKET_PORT)
    print("WebSocket server started. Waiting for Unity to connect...")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
