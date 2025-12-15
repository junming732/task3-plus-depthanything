import argparse
import cv2
import numpy as np
import torch
import os
import sys

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_src_path = os.path.join(current_dir, '../Depth-Anything-3/src')
sys.path.append(repo_src_path)

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("CRITICAL: Could not import library. Check 'ls Depth-Anything-3/src'")
    sys.exit(1)

def parse_cameras_txt(calib_path):
    """Parses ETH3D intrinsics (fx, fy, cx, cy)"""
    if not os.path.exists(calib_path): return None
    with open(calib_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            # ETH3D Pinhole: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
            if len(parts) >= 8 and parts[1] == 'PINHOLE':
                return float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
    return None

def save_ply(points, colors, filename):
    """Saves a high-res PLY file"""
    num_points = points.shape[0]
    print(f"Saving {num_points/1_000_000:.2f} MILLION points to {filename}...")

    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    # Stack data (N, 6)
    data = np.hstack([points, colors]).astype(object)

    # Save (This will be large, approx 500MB+)
    np.savetxt(filename, data, fmt="%.4f %.4f %.4f %d %d %d", header=header, comments="")
    print(f"Saved successfully.")

def reconstruct(image_path, output_path, model, calib_path=None):
    print(f"Loading High-Res Image: {image_path}")

    # 1. Load Original Image
    raw_image = cv2.imread(image_path)
    if raw_image is None: raise ValueError("Image load failed")

    # Get original dimensions (e.g. 6000 x 4000)
    h_orig, w_orig = raw_image.shape[:2]
    print(f"Original Resolution: {w_orig}x{h_orig}")

    # 2. Inference
    with torch.no_grad():
        pred = model.inference([raw_image])
        depth_small = pred.depth[0]

    # 3. UPSCALING (High Fidelity Mode)
    # We resize depth to match the HUGE original image
    print("Upscaling depth map to full resolution...")
    depth = cv2.resize(depth_small, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

    # 4. Intrinsics (Original)
    fx, fy = w_orig, w_orig
    cx, cy = w_orig/2, h_orig/2

    if calib_path:
        intrinsics = parse_cameras_txt(calib_path)
        if intrinsics:
            fx, fy, cx, cy = intrinsics
            print(f"Using Original Intrinsics: fx={fx:.2f}, cx={cx:.2f}")

    # 5. Back-projection
    # Generate grid for full resolution
    x = np.linspace(0, w_orig - 1, w_orig)
    y = np.linspace(0, h_orig - 1, h_orig)
    u, v = np.meshgrid(x, y)

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Flatten
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    # 6. Safety Downsample (Optional)
    # If the file is >1GB or RAM crashes, uncomment the next two lines:
    # points = points[::2]
    # colors = colors[::2]

    save_ply(points, colors, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--calib', type=str, default=None)
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = "depth-anything/DA3-LARGE"

    print(f"Loading Model: {MODEL_NAME}")
    model = DepthAnything3.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    reconstruct(args.img, args.out, model, args.calib)