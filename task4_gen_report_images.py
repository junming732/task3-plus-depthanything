import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import glob
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from PIL import Image

# --- CONFIGURATION ---
ROOT_DIR = "/home/junming/nobackup_junming/eth3d-dataset/train"
IMG_ROOT = os.path.join(ROOT_DIR, "undistorted")
GT_ROOT  = os.path.join(ROOT_DIR, "ground_truth")
NPZ_DIR  = "/home/junming/nobackup_junming/task4/training_data_dual"
MODEL_PATH = "scalenet_dual.pth"
OUTPUT_DIR = "report_images_v7" # Final Polish

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. MODEL DEFINITIONS ---
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("Error: Install DepthAnything3 first.")
    sys.exit(1)

class ScaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.mlp(x)

# --- 2. HELPER: Render 3-Panel (RGB | Depth | 3D Isometric) ---
def render_triplet(ply_path, img_path, depth_map, output_png, title="", scale_val=0.0):
    print(f"Rendering Triplet -> {output_png}...")

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 8))
    fig.patch.set_facecolor('#1e1e1e')

    # --- PANEL 1: RGB ---
    ax_rgb = fig.add_subplot(1, 3, 1)
    raw_img = cv2.imread(img_path)
    if raw_img is not None:
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        ax_rgb.imshow(raw_img)
    ax_rgb.set_title("1. Input Context (ResNet)", color='white', fontsize=14)
    ax_rgb.axis('off')

    # --- PANEL 2: DEPTH MAP ---
    ax_depth = fig.add_subplot(1, 3, 2)
    # Squeeze (1, H, W) -> (H, W)
    if len(depth_map.shape) == 3: depth_map = depth_map.squeeze()

    d_min, d_max = depth_map.min(), depth_map.max()
    d_norm = (depth_map - d_min) / (d_max - d_min + 1e-6)

    ax_depth.imshow(d_norm, cmap='magma')
    ax_depth.set_title("2. Relative Geometry (DA3)", color='white', fontsize=14)
    ax_depth.axis('off')

    # --- PANEL 3: 3D ISOMETRIC VIEW ---
    ax_3d = fig.add_subplot(1, 3, 3, projection='3d')
    ax_3d.set_facecolor('black')

    # Read PLY
    points, colors = [], []
    with open(ply_path, 'r') as f:
        header_ended = False
        for line in f:
            if not header_ended:
                if line.strip() == "end_header": header_ended = True
                continue
            vals = line.strip().split()
            if len(vals) >= 6:
                points.append([float(vals[0]), float(vals[1]), float(vals[2])])
                colors.append([int(vals[3])/255.0, int(vals[4])/255.0, int(vals[5])/255.0])

    points = np.array(points)
    colors = np.array(colors)

    if len(points) > 30000:
        idx = np.random.choice(len(points), 30000, replace=False)
        points = points[idx]
        colors = colors[idx]

    # Cleanup outliers
    z_med = np.median(points[:,2])
    z_mad = np.median(np.abs(points[:,2] - z_med))
    mask = np.abs(points[:,2] - z_med) < (4.0 * z_mad)
    points = points[mask]
    colors = colors[mask]

    # Center
    points -= np.mean(points, axis=0)

    # Invert Axes for visualization
    xs, ys, zs = points[:, 0], -points[:, 1], -points[:, 2] # Depth is Z

    ax_3d.scatter(xs, ys, zs, c=colors, s=1.5, marker='.', alpha=0.9)

    # Set Angle: "Chase Camera"
    # elev=15 (Slightly above floor)
    # azim=-170 (Looking almost straight down the hall)
    ax_3d.view_init(elev=15, azim=-110)

    # Tight Box
    max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (xs.max()+xs.min())*0.5, (ys.max()+ys.min())*0.5, (zs.max()+zs.min())*0.5
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

    ax_3d.set_xlabel("X (Width)", color='white')
    ax_3d.set_ylabel("Y (Height)", color='white')
    ax_3d.set_zlabel("Z (Depth)", color='white')

    ax_3d.set_title(f"3. Metric 3D (Scale={scale_val:.2f})", color='white', fontsize=14)
    ax_3d.grid(False) # Turn off grid for cleaner look
    ax_3d.axis('off')

    plt.subplots_adjust(wspace=0.05)
    plt.savefig(output_png, dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved {output_png}")

# --- 3. HELPER: Generate Data (Same) ---
def generate_data(da3, resnet, scalenet, img_path, out_ply):
    img_cv = cv2.imread(img_path)
    img_pil = Image.open(img_path).convert('RGB')
    resnet_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        pred = da3.inference(image=[img_cv])
        rel_depth = pred.depth if hasattr(pred, 'depth') else pred['depth']
        if isinstance(rel_depth, torch.Tensor): rel_depth = rel_depth.cpu().numpy()

        feat = resnet(resnet_transform(img_pil).unsqueeze(0).to(DEVICE)).squeeze()
        scale = scalenet(feat).item()

        metric_depth = rel_depth * scale
        if len(metric_depth.shape)==3: metric_depth = metric_depth.squeeze()
        H, W = metric_depth.shape
        rgb = cv2.cvtColor(cv2.resize(img_cv, (W,H)), cv2.COLOR_BGR2RGB).reshape(-1,3)
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        x = (x-W/2) * (metric_depth/1000.0)
        y = (y-H/2) * (metric_depth/1000.0)
        pts = np.stack((x,y,metric_depth), axis=-1).reshape(-1,3)
        mask = metric_depth.reshape(-1) > 0
        pts = pts[mask]; rgb = rgb[mask]

        with open(out_ply, "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for i in range(0, len(pts), 10):
                f.write(f"{pts[i][0]:.3f} {pts[i][1]:.3f} {pts[i][2]:.3f} {int(rgb[i][0])} {int(rgb[i][1])} {int(rgb[i][2])}\n")

    return scale, rel_depth

# --- 4. MAIN ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Setup
    print("Scanning Dataset...")
    valid_images = []
    scene_folders = sorted([f for f in os.listdir(IMG_ROOT) if os.path.isdir(os.path.join(IMG_ROOT, f))])
    for scene in scene_folders:
        img_dir = os.path.join(IMG_ROOT, scene, "images", "dslr_images_undistorted")
        gt_dir = os.path.join(GT_ROOT, scene)
        if not os.path.exists(img_dir): continue
        if not any(x.endswith('.ply') for r,d,f in os.walk(gt_dir) for x in f): continue
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.JPG")) + glob.glob(os.path.join(img_dir, "*.jpg")))
        valid_images.extend(imgs)

    npz_files = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
    if len(npz_files) != len(valid_images):
        limit = min(len(npz_files), len(valid_images))
        npz_files, valid_images = npz_files[:limit], valid_images[:limit]

    # Test Split
    split_idx = int(0.8 * len(npz_files))
    test_npzs, test_imgs = npz_files[split_idx:], valid_images[split_idx:]

    # 2. Load Models
    print("Loading Dual-Stream Models...")
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3-Large").to(DEVICE)
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE).eval()
    scalenet = ScaleNet().to(DEVICE)
    scalenet.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    scalenet.eval()

    # 3. Find Best/Worst
    results = []
    print("Evaluating...")
    for i, (npz, img_path) in enumerate(zip(test_npzs, test_imgs)):
        d = np.load(npz)
        feat = torch.from_numpy(d['feature']).float().to(DEVICE).unsqueeze(0)
        target = float(d['target_scale'])
        with torch.no_grad(): pred = scalenet(feat).item()
        error = abs(pred - target) / (target + 1e-6)
        results.append({"img_path": img_path, "error": error, "pred": pred, "target": target})

    results.sort(key=lambda x: x['error'])
    candidates = [("best", results[0]), ("worst", results[-1]), ("random", random.choice(results))]

    # 4. Generate
    for name, data in candidates:
        print(f"\nProcessing {name.upper()}...")
        ply_name = os.path.join(OUTPUT_DIR, f"{name}.ply")
        png_name = os.path.join(OUTPUT_DIR, f"{name}_final.png")

        # Get Data
        scale, rel_depth = generate_data(da3, resnet, scalenet, data['img_path'], ply_name)

        # Render
        title = f"{name.upper()} | Error: {data['error']:.2f}"
        render_triplet(ply_name, data['img_path'], rel_depth, png_name, title, scale)

if __name__ == "__main__":
    main()