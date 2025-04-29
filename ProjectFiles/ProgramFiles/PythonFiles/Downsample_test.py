import os
import open3d as o3d
import numpy as np

def downsample_ply(input_file, output_file, keep_ratio):  # Reduce keep ratio for more downsampling
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(input_file)
    
    # Convert to numpy array
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        print(f"Error: The input file '{input_file}' contains no points.")
        return
    
    # Randomly select a subset of points to keep
    num_points = len(points)
    num_keep = int(num_points * keep_ratio)
    indices = np.random.choice(num_points, num_keep, replace=False)
    
    # Filter the point cloud
    pcd.points = o3d.utility.Vector3dVector(points[indices])
    
    # Ensure changes are saved properly by overwriting the file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"Saved downsampled point cloud with {num_keep} points to {output_file}")

def main():
    input_file = "Skull.ply"
    output_file = "Goat_skull_downsampled.ply"
    keep_ratio = 0.03  # Decrease the keep ratio for more downsampling

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    downsample_ply(input_file, output_file, keep_ratio)
    
    # Remove the problematic line or replace it with a proper check if needed.
    # Assuming you want to show the point clouds without the condition, just visualize them:

    pcd = o3d.io.read_point_cloud(output_file)
    pco = o3d.io.read_point_cloud(input_file)

    o3d.visualization.draw_geometries([pco], window_name="Original Point Cloud")
    o3d.visualization.draw_geometries([pcd], window_name="Downsampled Point Cloud")

    
    

if __name__ == "__main__":
    main()