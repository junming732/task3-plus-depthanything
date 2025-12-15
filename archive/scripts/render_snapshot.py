import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def read_ply(filename):
    points = []
    colors = []
    with open(filename, 'r') as f:
        header_ended = False
        for line in f:
            if not header_ended:
                if line.strip() == "end_header": header_ended = True
                continue
            vals = line.strip().split()
            if len(vals) >= 6:
                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
                # Colors 0-255 -> 0.0-1.0
                colors.append([int(vals[3])/255.0, int(vals[4])/255.0, int(vals[5])/255.0])
    return np.array(points), np.array(colors)

def render_snapshot(ply_path, output_path):
    print(f"Loading {ply_path}...")
    points, colors = read_ply(ply_path)

    # 1. Subsample (Matplotlib limit)
    # Increase to 30k for density, but it will be slower
    target_points = 30000
    if len(points) > target_points:
        indices = np.random.choice(len(points), target_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    # --- AGGRESSIVE FILTERING (The Fix) ---
    # We use Median Absolute Deviation (MAD) to find the "blob" of the courtyard
    # and ignore the "sky" points that are kilometers away.
    z_median = np.median(points[:, 2])
    z_diff = np.abs(points[:, 2] - z_median)
    mad = np.median(z_diff)

    # Keep points within 3 deviations of the median depth
    # This keeps the building/ground and cuts the sky
    mask = z_diff < (3.0 * mad)

    points = points[mask]
    colors = colors[mask]
    print(f"Remaining points after filtering: {len(points)}")

    # Center the cloud
    points -= np.mean(points, axis=0)

    # --- PLOTTING "DARK MODE" ---
    plt.style.use('dark_background') # Make it look like the demo videos
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Setup Black Background
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.grid(False) # Turn off grid
    ax.set_axis_off() # Turn off axes

    # Invert axes for camera view
    xs = points[:, 0]
    ys = -points[:, 1]
    zs = -points[:, 2]

    # s=2.0 or s=3.0 makes the points "splat" and overlap
    # This creates the illusion of a solid surface
    ax.scatter(xs, ys, zs, c=colors, s=3.0, marker='o', alpha=0.8)

    # Force Tight Bounding Box
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Angle: Slightly up and rotated
    ax.view_init(elev=20, azim=-135)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved dense dark-mode snapshot to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply', type=str, required=True)
    parser.add_argument('--out', type=str, default="snapshot_dark.png")
    args = parser.parse_args()
    render_snapshot(args.ply, args.out)