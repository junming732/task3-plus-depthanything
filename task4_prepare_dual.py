import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import glob
import sys
from tqdm import tqdm
from PIL import Image

# --- 1. Imports ---
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("Error: Install DepthAnything3 first.")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. Configuration ---
ROOT_DIR = "/home/junming/nobackup_junming/eth3d-dataset/train"
IMG_ROOT = os.path.join(ROOT_DIR, "undistorted")
GT_ROOT  = os.path.join(ROOT_DIR, "ground_truth")

# New Clean Output Folder
OUTPUT_DIR = "/home/junming/nobackup_junming/task4/training_data_dual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3. Setup Models ---
print("--- Loading Models ---")

# A. Depth Stream (DA3) - The "Eyes"
da3_model = DepthAnything3.from_pretrained("depth-anything/DA3-Large").to(DEVICE)

# B. Context Stream (ResNet18) - The "Brain"
# We remove the last layer (fc) to get raw features (512 dims) instead of classes
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]) # Output: [B, 512, 1, 1]
feature_extractor = feature_extractor.to(DEVICE).eval()

# Standard ResNet Normalization
resnet_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 4. Helper: Parse PLY ---
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

# --- 5. Processing Loop ---
scene_folders = [f for f in os.listdir(IMG_ROOT) if os.path.isdir(os.path.join(IMG_ROOT, f))]
count = 0

print(f"Writing Dual-Stream data to: {OUTPUT_DIR}")

for scene in scene_folders:
    # Locate Images
    img_dir = os.path.join(IMG_ROOT, scene, "images", "dslr_images_undistorted")
    if not os.path.exists(img_dir): continue
    images = sorted(glob.glob(os.path.join(img_dir, "*.JPG")) + glob.glob(os.path.join(img_dir, "*.jpg")))
    if not images: continue

    # Locate GT
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
        # 1. Load Image (OpenCV for DA3, PIL for ResNet)
        raw_cv = cv2.imread(img_path)
        raw_pil = Image.open(img_path).convert('RGB')

        if raw_cv is None: continue

        with torch.no_grad():
            # STREAM A: Get Depth (DA3)
            pred = da3_model.inference(image=[raw_cv])
            if hasattr(pred, 'depth'): rel_depth = pred.depth
            elif isinstance(pred, dict): rel_depth = pred['depth']
            else: rel_depth = pred
            if isinstance(rel_depth, torch.Tensor): rel_depth = rel_depth.cpu().numpy()

            # STREAM B: Get Features (ResNet)
            # Guaranteed to work because it's standard Torchvision
            input_tensor = resnet_preprocess(raw_pil).unsqueeze(0).to(DEVICE)
            feat_vector = feature_extractor(input_tensor).squeeze().cpu().numpy() # [512]

            # Calc Scale
            rel_median = np.nanmedian(rel_depth)
            scale_factor_needed = target_metric_scale / (rel_median + 1e-6)

            # Save
            np.savez(os.path.join(OUTPUT_DIR, f"train_{count:04d}.npz"),
                     feature=feat_vector,
                     target_scale=scale_factor_needed)
            count += 1

print(f"Done. Prepared {count} VALID samples.")