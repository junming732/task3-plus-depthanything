import torch
import cv2
import numpy as np
import os
import glob
import sys
from tqdm import tqdm

# --- 1. Import ---
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("Error: Install DepthAnything3 first.")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. Configuration (User Requested Path) ---
# Input Data (ETH3D)
ROOT_DIR = "/home/junming/nobackup_junming/eth3d-dataset/train"
IMG_ROOT = os.path.join(ROOT_DIR, "undistorted")
GT_ROOT  = os.path.join(ROOT_DIR, "ground_truth")

# Output Data (Your Custom Path)
OUTPUT_DIR = "/home/junming/nobackup_junming/task4/training_data_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3. Helper: Parse PLY for Median Depth ---
def get_scene_metric_scale(ply_path):
    try:
        with open(ply_path, 'rb') as f:
            while True:
                line = f.readline().strip()
                if line == b'end_header': break
                if not line: return 10.0
            data = np.fromfile(f, dtype=np.float32)
            valid_data = data[::100]
            valid_data = valid_data[(valid_data > 0.1) & (valid_data < 500)]
            if len(valid_data) == 0: return 10.0
            return float(np.nanmedian(valid_data))
    except:
        return 10.0

# --- 4. Load Model ---
print(f"--- Loading DA3 on {DEVICE} ---")
model_api = DepthAnything3.from_pretrained("depth-anything/DA3-Large").to(DEVICE)

# --- 5. THE FIX: Robust Hook ---
global_features = {}

def hook_fn(module, input, output):
    # Unwrap List/Tuple (The Bug Fix)
    if isinstance(output, (list, tuple)): feat = output[-1]
    else: feat = output

    # Unwrap Dict/Object
    if hasattr(feat, 'x_norm_clstoken'): feat = feat.x_norm_clstoken
    elif isinstance(feat, dict): feat = feat.get('x_norm_clstoken', feat.get('prediction'))

    # Pool
    if hasattr(feat, 'shape'):
        if len(feat.shape) == 4: # [B, C, H, W]
            global_features['feat'] = feat.mean(dim=[2, 3]).detach().cpu().numpy()
        elif len(feat.shape) == 3: # [B, N, C]
            global_features['feat'] = feat.mean(dim=1).detach().cpu().numpy()
        elif len(feat.shape) == 2: # [B, C]
            global_features['feat'] = feat.detach().cpu().numpy()

# Attach Hook
if hasattr(model_api, 'model') and hasattr(model_api.model, 'backbone'):
    model_api.model.backbone.register_forward_hook(hook_fn)

# --- 6. Processing ---
scene_folders = [f for f in os.listdir(IMG_ROOT) if os.path.isdir(os.path.join(IMG_ROOT, f))]
count = 0

print(f"Writing data to: {OUTPUT_DIR}")

for scene in scene_folders:
    # Find Images
    img_dir = os.path.join(IMG_ROOT, scene, "images", "dslr_images_undistorted")
    if not os.path.exists(img_dir): continue
    images = sorted(glob.glob(os.path.join(img_dir, "*.JPG")) + glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images: continue

    # Find GT
    gt_scene_dir = os.path.join(GT_ROOT, scene)
    ply_files = []
    for r, d, f in os.walk(gt_scene_dir):
        for file in f:
            if file.endswith('.ply'): ply_files.append(os.path.join(r, file))
    if not ply_files: continue

    # Get Target
    target_metric_scale = get_scene_metric_scale(ply_files[0])
    if target_metric_scale < 0.5: target_metric_scale = 10.0

    print(f"Processing {scene} (Target: {target_metric_scale:.2f}m)...")

    for img_path in tqdm(images):
        raw_img = cv2.imread(img_path)
        if raw_img is None: continue

        with torch.no_grad():
            pred = model_api.inference(image=[raw_img])
            if hasattr(pred, 'depth'): rel_depth = pred.depth
            elif isinstance(pred, dict): rel_depth = pred['depth']
            else: rel_depth = pred
            if isinstance(rel_depth, torch.Tensor): rel_depth = rel_depth.cpu().numpy()

            # CRITICAL: Only save if hook worked
            if 'feat' not in global_features: continue
            feat_vector = global_features['feat']

            # Check for zeros (The final check)
            if np.all(feat_vector == 0): continue

            # Calc Scale Factor
            rel_median = np.nanmedian(rel_depth)
            scale_factor_needed = target_metric_scale / (rel_median + 1e-6)

            save_path = os.path.join(OUTPUT_DIR, f"train_{count:04d}.npz")
            np.savez(save_path,
                     feature=feat_vector,
                     target_scale=scale_factor_needed)
            count += 1

print(f"Done. Prepared {count} VALID samples in {OUTPUT_DIR}")