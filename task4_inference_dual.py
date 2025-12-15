import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import argparse
import sys
import os
from PIL import Image

# --- 1. Import ---
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("Error: Install DepthAnything3 first.")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. Dual-Stream Architecture ---
class ScaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches your 90% accuracy model
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.mlp(x)

# --- 3. PLY Saver ---
def save_ply_manual(image, depth_map, output_path, focal_length=1000.0):
    print(f"Generating 3D Cloud -> {output_path}...")

    if len(depth_map.shape) == 3: depth_map = depth_map.squeeze()
    H, W = depth_map.shape

    # Resize RGB to match Depth
    rgb_resized = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_map
    x = (x_grid - W / 2) * (z / focal_length)
    y = (y_grid - H / 2) * (z / focal_length)

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
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
    print("Saved successfully.")

# --- 4. Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="scalenet_dual.pth")
    parser.add_argument("--out", type=str, default="final_result_dual.ply")
    args = parser.parse_args()

    # A. Load Depth Stream (DA3)
    print("Loading Stream A: Depth Anything V3...")
    da3_model = DepthAnything3.from_pretrained("depth-anything/DA3-Large").to(DEVICE)

    # B. Load Context Stream (ResNet18)
    print("Loading Stream B: ResNet18...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE).eval()

    resnet_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # C. Load ScaleNet
    print("Loading ScaleNet Head...")
    if not os.path.exists(args.weights):
        print(f"Error: {args.weights} not found.")
        sys.exit(1)
    scalenet = ScaleNet().to(DEVICE)
    scalenet.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    scalenet.eval()

    # D. Process
    img_cv = cv2.imread(args.img)
    img_pil = Image.open(args.img).convert('RGB')
    if img_cv is None: sys.exit(1)

    with torch.no_grad():
        # 1. Get Relative Depth
        pred = da3_model.inference(image=[img_cv])
        if hasattr(pred, 'depth'): rel_depth = pred.depth
        elif isinstance(pred, dict): rel_depth = pred['depth']
        else: rel_depth = pred
        if isinstance(rel_depth, torch.Tensor): rel_depth = rel_depth.cpu().numpy()

        # 2. Get Features & Scale
        input_tensor = resnet_preprocess(img_pil).unsqueeze(0).to(DEVICE)
        feat = feature_extractor(input_tensor).squeeze() # [512]

        if len(feat.shape) == 1: feat = feat.unsqueeze(0)
        scale_factor = scalenet(feat).item()

        print(f"--- RESULTS ---")
        print(f"Predicted Metric Scale: {scale_factor:.4f}")

        # 3. Apply & Save
        metric_depth = rel_depth * scale_factor
        save_ply_manual(img_cv, metric_depth, args.out)

if __name__ == "__main__":
    main()