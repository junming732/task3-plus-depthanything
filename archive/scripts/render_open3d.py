import open3d as o3d
import numpy as np
import argparse
import os

# FORCE HEADLESS MODE
os.environ['EGL_PLATFORM'] = 'surfaceless'

def render_headless(ply_path, output_prefix):
    print(f"Loading {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)

    # 1. AGGRESSIVE CLEANING
    # Use tighter threshold (0.5) to delete ALL sky/noise
    print("Cleaning noise...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.5)
    pcd = pcd.select_by_index(ind)

    # 2. CENTER THE DATA
    center = pcd.get_center()
    pcd.translate(-center) # Move to (0,0,0)

    # 3. CALCULATE ZOOM
    bbox = pcd.get_axis_aligned_bounding_box()
    max_dim = max(bbox.get_max_bound() - bbox.get_min_bound())
    dist = max_dim * 1.2  # Distance to place camera

    # 4. RENDERER SETUP
    width = 1920
    height = 1080
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 6.0 # Very thick points

    render.scene.add_geometry("pcd", pcd, mat)

    # 5. GENERATE 4 VIEWS (The Fix)
    # We define 4 positions to look from
    views = {
        "front": ([0, 0, dist], [0, 1, 0]),   # Look from Z, Up is Y
        "top":   ([0, dist, 0], [0, 0, -1]),  # Look from Y, Up is -Z
        "side":  ([dist, 0, 0], [0, 1, 0]),   # Look from X, Up is Y
        "iso":   ([dist, -dist, dist], [0, 1, 0]) # Isometric Corner
    }

    for name, (eye, up) in views.items():
        # Look at (0,0,0) from 'eye' with 'up' vector
        render.scene.camera.look_at([0,0,0], np.array(eye), np.array(up))

        img = render.render_to_image()
        filename = f"{output_prefix}_{name}.png"
        o3d.io.write_image(filename, img, 9)
        print(f"Saved {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply', type=str, required=True)
    parser.add_argument('--out', type=str, default="outputs/render")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    render_headless(args.ply, args.out)