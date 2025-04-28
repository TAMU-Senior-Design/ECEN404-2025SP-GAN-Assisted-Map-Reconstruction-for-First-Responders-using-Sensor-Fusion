import os
import pyrealsense2 as rs
import numpy as np
import cv2

# Create directories for RGB and Depth images
rgb_dir = 'rgb_images_testing'
depth_dir = 'depth_images_testing'

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# Initialize the realsense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the pipeline to stream RGB and depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

try:
    # Frame counter for saving unique filenames
    frame_count = 0

    while True:
        # Wait for a frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Ensure both frames have been captured
        if not color_frame or not depth_frame:
            continue

        # Convert to numpy arrays
        rgb_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply colormap on depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = rgb_image.shape

        # If depth and color resolutions differ, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(rgb_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((rgb_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)

        # Wait for a key press
        key = cv2.waitKey(1)

        if key == ord('s'):  # If the 's' key is pressed
            # Save the current frames
            rgb_path = f'{rgb_dir}/rgb_{frame_count}.png'
            depth_path = f'{depth_dir}/depth_{frame_count}.png'
            cv2.imwrite(rgb_path, rgb_image)
            cv2.imwrite(depth_path, depth_colormap)
            print(f'Saved frame {frame_count}: RGB -> {rgb_path}, Depth -> {depth_path}')
            frame_count += 1

        elif key == ord('q'):  # If the 'q' key is pressed
            print("Exiting program...")
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
