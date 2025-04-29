import torch
import numpy as np
import cv2
from PIL import Image
import open3d as o3d
from CapstoneProject import Generator

# Load the trained generator model
def load_generator_model(model_path="generator.pth", img_dim=256, point_dim=3, num_points=1024):
    """
    Load the trained generator model with specified dimensions.
    """
    generator = Generator(img_dim=img_dim, point_dim=point_dim, num_points=num_points)
    generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    generator.eval()  # Set to evaluation mode
    return generator

def capture_image_from_camera():
    """
    Capture an image from the camera and return it as a numpy array (BGR format).
    
    Returns:
        numpy.ndarray: The captured image in BGR format.
    """
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise Exception("Could not open camera. Make sure a camera is connected and accessible.")

    print("Press 's' to capture an image or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        cv2.imshow("Camera Feed", frame)  # Show live feed

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to save the frame
            captured_image = frame  # Keep it as color (BGR)
            break
        elif key == ord('q'):  # Press 'q' to quit
            captured_image = None
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_image is not None:
        print("Image captured successfully!")
    else:
        print("No image captured.")

    return captured_image


# def generate_point_cloud_from_image(image):
#     """
#     Generate a point cloud from an input image.
    
#     Parameters:
#         image (numpy.ndarray): Grayscale image array.
        
#     Returns:
#         numpy.ndarray: Nx3 array of points representing the point cloud.
#     """
#     # Get the image dimensions
#     height, width = image.shape

#     # Generate point cloud
#     points = []
#     for y in range(height):
#         for x in range(width):
#             intensity = image[y, x]
#             points.append([x, y, intensity])  # x, y, and pixel intensity as z

#     return np.array(points)

import numpy as np
import cv2
import open3d as o3d

def generate_point_cloud_from_image(color_image):
    """
    Generate a 3D point cloud from an RGB image with estimated depth.

    Parameters:
        color_image (numpy.ndarray): The original color image.

    Returns:
        numpy.ndarray: Nx6 array containing (x, y, z, r, g, b) points.
    """
    # Get the image dimensions
    height, width, _ = color_image.shape
    
    # Convert to grayscale for depth estimation
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Normalize the grayscale values to range [0, 1]
    normalized_depth = grayscale_image.astype(np.float32) / 255.0

    # Scale depth values (Adjust scale factor if needed)
    depth_scale = 100  # Modify this to control depth range
    depth_map = normalized_depth * depth_scale  

    points = []

    for y in range(height):
        for x in range(width):
            z = depth_map[y, x]  # Properly scaled depth
            r, g, b = color_image[y, x]  # Extract original RGB values
            
            # Invert Y-axis for correct orientation in 3D space
            points.append([x, height - y, z, r, g, b])  

    return np.array(points, dtype=np.float32)  # Ensure float32 for compatibility




# def save_point_cloud_to_ply(points, filename="output_point_cloud.ply"):
#     """
#     Save a point cloud to a PLY file manually.

#     Parameters:
#         points (numpy.ndarray): Nx3 array of points.
#         filename (str): Output PLY file name.
#     """
#     num_points = points.shape[0]
#     header = f"""ply
# format ascii 1.0
# element vertex {num_points}
# property float x
# property float y
# property float z
# end_header
# """
#     with open(filename, 'w') as file:
#         file.write(header)
#         np.savetxt(file, points, fmt="%.6f")
#     print(f"Point cloud saved to {filename}")

# def save_point_cloud_to_obj(points, filename="output_point_cloud.obj"):
#     """
#     Save a point cloud to an OBJ file manually.

#     Parameters:
#         points (numpy.ndarray): Nx3 array of points.
#         filename (str): Output OBJ file name.
#     """
#     num_points = points.shape[0]
    
#     with open(filename, 'w') as file:
#         # Write OBJ file header
#         file.write("# Point Cloud\n")
        
#         # Write vertices
#         for point in points:
#             file.write(f"v {point[0]} {point[1]} {point[2]}\n")
        
#     print(f"Point cloud saved to {filename}")

def save_point_cloud_to_obj(points, filename="output_point_cloud.obj"):
    """
    Save a point cloud to an OBJ file with vertex colors.

    Parameters:
        points (numpy.ndarray): Nx6 array of points (x, y, z, r, g, b).
        filename (str): Output OBJ file name.
    """
    with open(filename, 'w') as file:
        file.write("# OBJ file with point cloud and RGB colors\n")
        for point in points:
            x, y, z, r, g, b = point
            r, g, b = r / 255.0, g / 255.0, b / 255.0  # Normalize colors (0-1)
            file.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.3f} {g:.3f} {b:.3f}\n")

    print(f"Point cloud saved as OBJ with RGB colors: {filename}")


# def load_and_visualize_obj(filename):
#     """
#     Load and visualize a PLY file using Open3D.
    
#     Parameters:
#         filename (str): Path to the PLY file.
#     """
#     try:
#         pcd = o3d.io.read_point_cloud(filename)
#         print(pcd)  # Print basic information about the point cloud
#         o3d.visualization.draw_geometries([pcd])  # Open the visualization window
#     except Exception as e:
#         print(f"Failed to load or visualize PLY file: {e}")

def load_and_visualize_obj(filename):
    """
    Load and visualize an OBJ file with colors using Open3D.

    Parameters:
        filename (str): Path to the OBJ file.
    """
    try:
        mesh = o3d.io.read_triangle_mesh(filename, enable_post_processing=True)
        if mesh.has_vertex_colors():
            print("Loaded OBJ with vertex colors successfully!")
        else:
            print("Warning: No vertex colors found. Viewer may not render them.")

        o3d.visualization.draw_geometries([mesh])

    except Exception as e:
        print(f"Failed to load or visualize OBJ file: {e}")


# Example usage
# if __name__ == "__main__":
#     # Step 1: Capture an image from the camera
#     image = capture_image_from_camera()
    
#     if image is not None:
#         # Step 2: Generate point cloud from the captured image
#         points = generate_point_cloud_from_image(image)
        
#         # Step 3: Save point cloud to a obj file
#         save_point_cloud_to_obj(points, "output_point_cloud.obj")
        
#         # Step 4: Load and visualize the obj file
#         load_and_visualize_obj("output_point_cloud.obj")


# Main execution
if __name__ == "__main__":
    # Step 1: Capture an image from the camera
    color_image = capture_image_from_camera()  # Your existing function

    if color_image is not None:
        # Step 2: Generate the point cloud
        points = generate_point_cloud_from_image(color_image)

        # Step 3: Save as an OBJ file with RGB colors
        save_point_cloud_to_obj(points, "output_point_cloud_with_color.obj")

        print("OBJ file successfully saved with correct RGB colors!")


