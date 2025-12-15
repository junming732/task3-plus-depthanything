import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
import sys
import os

# --- 1. Import ---
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("Error: Install DepthAnything3 first.")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. ScaleNet Definition ---
class ScaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.mlp(x)

# --- 3. PLY Saver ---
def save_ply_manual(image, depth_map, output_path, focal_length=1000.0):
    print(f"Generating 3D Cloud -> {output_path}...")
    H, W = depth_map.shape
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))

    z = depth_map
    x = (x_grid - W / 2) * (z / focal_length)
    y = (y_grid - H / 2) * (z / focal_length)

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    mask = (z.reshape(-1) > 0)
    points = points[mask]
    rgb = rgb[mask]

    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(output_path, "w") as f:
        f.write(header)
        for i in range(len(points)):
            p = points[i]
            c = rgb[i]
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    print("Saved.")

# --- 4. Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="scalenet_best.pth", help="Path to trained model")
    parser.add_argument("--out", type=str, default="final_result.ply", help="Output filename")
    args = parser.parse_args()

    # Load Model
    print("Loading Depth Anything V3...")
    model_api = DepthAnything3.from_pretrained("depth-anything/DA3-Large").to(DEVICE)

    # --- ROBUST HOOK LOGIC ---
    global_features = {}

    def hook_fn(module, input, output):
        print("DEBUG: Hook Triggered!") # Confirm it runs

        # Robust extraction
        feat = output
        if hasattr(output, 'x_norm_clstoken'): feat = output.x_norm_clstoken
        elif isinstance(output, dict): feat = output.get('x_norm_clstoken', output.get('prediction'))
        elif isinstance(output, (tuple, list)): feat = output[-1]

        print(f"DEBUG: Hook output type: {type(feat)}")
        if hasattr(feat, 'shape'):
            print(f"DEBUG: Hook output shape: {feat.shape}")

            # Handle different shapes [B, C, H, W] vs [B, N, C]
            if len(feat.shape) == 4:
                global_features['feat'] = feat.mean(dim=[2, 3]).detach().cpu()
            elif len(feat.shape) == 3:
                global_features['feat'] = feat.mean(dim=1).detach().cpu()
            elif len(feat.shape) == 2:
                # Already pooled? [B, C]
                global_features['feat'] = feat.detach().cpu()

    # Attach Hook
    attached = False
    if hasattr(model_api, 'model'):
        if hasattr(model_api.model, 'backbone'):
            model_api.model.backbone.register_forward_hook(hook_fn)
            print("DEBUG: Attached to model_api.model.backbone")
            attached = True
        elif hasattr(model_api.model, 'encoder'):
            model_api.model.encoder.register_forward_hook(hook_fn)
            print("DEBUG: Attached to model_api.model.encoder")
            attached = True

    if not attached:
        print("WARNING: Could not find backbone/encoder! Printing model structure:")
        print(model_api)

    # Load ScaleNet
    print("Loading ScaleNet...")
    if not os.path.exists(args.weights):
        print(f"Error: Weights {args.weights} not found.")
        sys.exit(1)

    scalenet = ScaleNet().to(DEVICE)
    scalenet.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    scalenet.eval()

    # Run
    img = cv2.imread(args.img)
    if img is None:
        print("Error reading image.")
        sys.exit(1)

    with torch.no_grad():
        # 1. Inference
        pred = model_api.inference(image=[img])

        # 2. Extract Depth
        if hasattr(pred, 'depth'): rel_depth = pred.depth
        elif isinstance(pred, dict): rel_depth = pred['depth']
        else: rel_depth = pred

        if isinstance(rel_depth, torch.Tensor): rel_depth = rel_depth.cpu().numpy()

        # 3. Check Hook
        if 'feat' not in global_features:
            print("CRITICAL ERROR: Hook did not capture features.")
            print("Trying fallback: Using dummy scale 10.0 (for debugging)")
            scale_factor = 10.0
        else:
            feat = global_features['feat'].to(DEVICE)
            if len(feat.shape) == 1: feat = feat.unsqueeze(0) # [1024] -> [1, 1024]

            # Predict
            scale_factor = scalenet(feat).item()
            print(f"--- SUCCESS ---")
            print(f"Network Predicted Scale: {scale_factor:.4f}")

        # 4. Save
        metric_depth = rel_depth * scale_factor
        save_ply_manual(img, metric_depth, args.out)

if __name__ == "__main__":
    main()