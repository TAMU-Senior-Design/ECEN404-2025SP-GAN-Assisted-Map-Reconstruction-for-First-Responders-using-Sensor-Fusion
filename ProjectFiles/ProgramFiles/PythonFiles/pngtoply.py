import numpy as np
import cv2
import open3d as o3d

def create_point_cloud(rgb_file, depth_file, fx, fy, cx, cy, save_to_ply=False, ply_filename="output.ply"):
    # Load RGB image and depth map
    rgb = cv2.imread(rgb_file)
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

    # Validate loaded images
    if rgb is None or depth is None:
        print("Error: Could not load images.")
        return

    # Rotate images 180 degrees
    rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    depth = cv2.rotate(depth, cv2.ROTATE_180)

    # Mirror images horizontally
    rgb = cv2.flip(rgb, 1)
    depth = cv2.flip(depth, 1)

    # Handle cases where depth has multiple channels
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    # Convert depth to meters
    if np.max(depth) > 100:
        depth = depth / 1000.0

    # Ensure nonzero depth values for point cloud generation
    depth[depth == 0] = np.nan

    # Create point cloud from RGB and depth images
    height, width = depth.shape
    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            z = depth[v, u]
            if np.isnan(z) or z <= 0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
            colors.append(rgb[v, u] / 255.0)  # Convert to [0, 1] for Open3D

    # Create Open3D point cloud
    if len(points) == 0:
        print("Warning: No valid points found in the depth image.")
        return

    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize point cloud
    o3d.visualization.draw_geometries([point_cloud])

    # Save point cloud to a PLY file
    if save_to_ply:
        o3d.io.write_point_cloud(ply_filename, point_cloud)
        print(f"Point cloud saved to {ply_filename}")


if __name__ == "__main__":
    # Camera intrinsics
    fx = 423.3  # Focal length in x axis
    fy = 423.3  # Focal length in y axis
    cx = 430.27  # Principal point in x axis
    cy = 236.9  # Principal point in y axis

    # RGB and depth image file paths
    rgb_file = "rgb_images_moms/rgb_287.png"
    depth_file = "depth_images_moms/depth_287.png"

    # Create and visualize point cloud
    create_point_cloud(rgb_file, depth_file, fx, fy, cx, cy, save_to_ply=True)