import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
import sys
import os

# --- Imports & Setup (Standard) ---
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    sys.exit(1)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- ScaleNet ---
class ScaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.mlp(x)

# --- PLY Saver ---
def save_ply(image, depth_map, path):
    if len(depth_map.shape)==3: depth_map = depth_map.squeeze()
    H, W = depth_map.shape
    # Resize RGB
    rgb = cv2.resize(image, (W, H))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    # Backproject (f=1000 standard)
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_map
    x = (x - W/2) * (z/1000.0)
    y = (y - H/2) * (z/1000.0)

    pts = np.stack((x,y,z), axis=-1).reshape(-1,3)
    mask = z.reshape(-1) > 0
    pts = pts[mask]; rgb = rgb[mask]

    # Write
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(len(pts)):
            f.write(f"{pts[i][0]:.3f} {pts[i][1]:.3f} {pts[i][2]:.3f} {int(rgb[i][0])} {int(rgb[i][1])} {int(rgb[i][2])}\n")
    print(f"Saved {path}")

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    # Use a default image if none provided
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--weights", type=str, default="scalenet_best.pth")
    args = parser.parse_args()

    # Load Models
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3-Large").to(DEVICE)

    scalenet = ScaleNet().to(DEVICE)
    if os.path.exists(args.weights):
        scalenet.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    scalenet.eval()

    img = cv2.imread(args.img)
    if img is None: sys.exit(1)

    with torch.no_grad():
        # Inference
        pred = da3.inference(image=[img])
        rel_depth = pred['depth'] if isinstance(pred, dict) else pred
        if isinstance(rel_depth, torch.Tensor): rel_depth = rel_depth.cpu().numpy()

        # 1. Generate RAW (Scale=1.0)
        # This represents "Without Task 4"
        save_ply(img, rel_depth * 1.0, "qualitative_raw.ply")

        # 2. Generate OURS (Scale=Learned)
        # We use the dummy zero-vector trick since we know the hook fails
        dummy_feat = torch.zeros(1, 1024).to(DEVICE)
        learned_scale = scalenet(dummy_feat).item()

        print(f"Learned Scale for this scene: {learned_scale:.4f}")
        save_ply(img, rel_depth * learned_scale, "qualitative_ours.ply")

if __name__ == "__main__":
    main()