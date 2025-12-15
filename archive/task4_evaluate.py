import torch
import numpy as np
import glob
import os
import torch.nn as nn
from torch.utils.data import Dataset

# --- 1. Define Model (Must match training) ---
class ScaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.mlp(x)

# --- 2. Load Data ---
class ScaleDataset(Dataset):
    def __init__(self, dir_path):
        self.files = sorted(glob.glob(os.path.join(dir_path, "*.npz")))
    def __len__(self): return len(self.files)

def evaluate():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Setup Data
    data_dir = "task4_training_data"
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

    if len(all_files) == 0:
        print(f"Error: No data in {data_dir}. Did you run prepare_data?")
        return

    # Split: 80% Train, 20% Test
    split = int(0.8 * len(all_files))
    train_files = all_files[:split]
    test_files  = all_files[split:]

    print(f"--- QUANTITATIVE EVALUATION ---")
    print(f"Train Set: {len(train_files)} samples")
    print(f"Test Set:  {len(test_files)} samples")

    # 2. Calculate Baseline (Naive Average)
    # "If I just guessed the average size of a scene, how wrong would I be?"
    train_targets = []
    for f in train_files:
        d = np.load(f)
        train_targets.append(float(d['target_scale']))

    baseline_guess = np.mean(train_targets)
    print(f"Baseline Guess (Training Mean): {baseline_guess:.4f}")

    # 3. Load Your Trained Model
    model = ScaleNet().to(DEVICE)
    if os.path.exists("scalenet_best.pth"):
        model.load_state_dict(torch.load("scalenet_best.pth", map_location=DEVICE))
    else:
        print("Error: scalenet_best.pth not found.")
        return
    model.eval()

    # 4. Run Comparison
    errors_baseline = []
    errors_ours = []

    print("\n[Test Set Results]")
    print(f"{'Sample':<10} | {'True':<8} | {'BaseErr':<8} | {'OursErr':<8} | {'Result'}")
    print("-" * 60)

    for f in test_files:
        d = np.load(f)
        feat = torch.from_numpy(d['feature']).float().to(DEVICE)
        target = float(d['target_scale'])

        # Handle the "Missing Hook" case (Zero Feature)
        # If feature is zero, model predicts the "Learned Bias"
        if len(feat.shape) == 0 or feat.sum() == 0:
            feat = torch.zeros(1, 1024).to(DEVICE)
        else:
            feat = feat.unsqueeze(0)

        with torch.no_grad():
            pred = model(feat).item()

        # Calculate Absolute Relative Error
        # err = |pred - target| / target
        e_base = abs(baseline_guess - target) / (target + 1e-6)
        e_ours = abs(pred - target) / (target + 1e-6)

        errors_baseline.append(e_base)
        errors_ours.append(e_ours)

        # Print first few
        if len(errors_ours) <= 5:
            winner = "Ours" if e_ours < e_base else "Base"
            print(f"{os.path.basename(f).replace('.npz',''):<10} | {target:.2f}     | {e_base:.2f}     | {e_ours:.2f}     | {winner}")

    # 5. Final Statistics
    mean_base_err = np.mean(errors_baseline)
    mean_ours_err = np.mean(errors_ours)
    improvement = ((mean_base_err - mean_ours_err) / mean_base_err) * 100

    print("-" * 60)
    print(f"FINAL METRICS (AbsRel Error):")
    print(f"Baseline Naive Error:  {mean_base_err:.4f}")
    print(f"ScaleNet (Ours) Error: {mean_ours_err:.4f}")
    print(f"Improvement:           {improvement:.2f}%")

if __name__ == "__main__":
    evaluate()