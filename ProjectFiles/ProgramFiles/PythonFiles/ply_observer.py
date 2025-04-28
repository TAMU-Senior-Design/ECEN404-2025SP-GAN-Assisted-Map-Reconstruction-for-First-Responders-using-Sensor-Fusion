import open3d as o3d

# Read the PLY file
mesh = o3d.io.read_triangle_mesh("output.ply")

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])