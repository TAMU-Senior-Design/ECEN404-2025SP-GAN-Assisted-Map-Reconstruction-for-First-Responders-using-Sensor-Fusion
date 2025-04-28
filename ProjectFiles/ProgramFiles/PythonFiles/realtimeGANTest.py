import os
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import asyncio
import websockets
import json
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import time

# Constant values (pls dont forget to put the downsampling factor here too)
INITIAL_DELAY_SECONDS = 2.0
WEBSOCKET_PORT = 8765
GAN_MODEL_PATH = "gan_model.pth"

# Generator architecture
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
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

# Initialize RealSense camera
def setup_realsense():
    print("Initializing RealSense pipeline...")

    # Start and configure pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure streams (640x480, 30 FPS)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"RealSense depth scale: {depth_scale}")

    # Wait for the camera to warm up so that first frame isnt garbage
    print(f"Waiting {INITIAL_DELAY_SECONDS} seconds for camera to stabilize...")
    time.sleep(INITIAL_DELAY_SECONDS)

    return pipeline, depth_scale

# Post-process depth frame
def clean_up(depth_frame):
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)

    return depth_frame

# Capture one frame of depth and color
def capture_and_process(pipeline):
    frames = pipeline.wait_for_frames()

    align_to = rs.stream.color
    align = rs.align(align_to)
    aligned = align.process(frames)

    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None
    
    depth_frame = clean_up(depth_frame)
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return depth_image, color_image

async def stream_to_unity(websocket):
    print("Connected to Unity.")
    # Setup camera
    pipeline, depth_scale = setup_realsense()

    # Load GAN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(GAN_MODEL_PATH, map_location=device))
    generator.eval()

    # Transformation for GAN input
    to_tensor = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
    ])

    # Camera intrinsics (DONT CHANGE THIS)
    fx, fy = 383.40667724609375, 383.40667724609375
    cx, cy = 325.6827087402344, 237.20228576660156

    while True:
        depth_image, color_image = capture_and_process(pipeline)
        if depth_image is None:
            print("Warning: No valid frames received from RealSense.")
            await asyncio.sleep(0.05)
            continue

        # Print image shapes for debugging
        print(f"depth_image shape: {depth_image.shape}, dtype: {depth_image.dtype}")
        print(f"color_image shape: {color_image.shape}, dtype: {color_image.dtype}")

        # Prepare input for GAN, convert BGR to RGB
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        input_tensor = to_tensor(rgb).unsqueeze(0).to(device)

        # Predict depth map via GAN
        with torch.no_grad():
            pred_depth = generator(input_tensor) 
        pred_depth_np = pred_depth.squeeze().cpu().numpy()
        depth_m = pred_depth_np 

        # Build point cloud
        h, w = depth_m.shape

        sampling_rate = 4
        print(f"Building point cloud of size {h}x{w} with sampling rate {sampling_rate}...")

        point_cloud = []
        valid_points = 0
        for y in range(0, h, sampling_rate):
            for x in range(0, w, sampling_rate):
                d = float(depth_m[y, x])

                if d <= 0:
                    continue

                # Convert depth to 3D coordinates
                X = (x - cx) * d / fx
                Y = (y - cy) * d / fy
                Z = d
                X_rot, Y_rot, Z_rot = -X, -Y, -Z

                b, g, r = color_image[y, x]

                # Append the 3D point with its color
                point_cloud.append([
                    X_rot, 
                    Y_rot, 
                    Z_rot, 
                    r/255.0, 
                    g/255.0, 
                    b/255.0
                ])
                valid_points += 1

        print(f"Collected {valid_points} points in point_cloud after downsampling.")
        for i in range(min(5, valid_points)):
            print(f"Sample point {i}: {point_cloud[i]}")

        # Convert the point cloud to JSON and send it
        try:
            msg = json.dumps(point_cloud)
            print(f"Sending point cloud JSON to Unity (length: {len(msg)} characters).")
            await websocket.send(msg)
        except Exception as e:
            print(f"Error converting point cloud to JSON: {e}")

        # Increase delay to reduce update frequency
        await asyncio.sleep(4.0)

async def main():
    print(f"Starting WebSocket server on ws://localhost:{WEBSOCKET_PORT}")
    server = await websockets.serve(stream_to_unity, "localhost", WEBSOCKET_PORT)
    print("WebSocket server started. Waiting for Unity to connect...")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
    print("WebSocket server stopped.")