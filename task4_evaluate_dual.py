import torch
import numpy as np
import glob
import os
import torch.nn as nn
from torch.utils.data import Dataset

# Must match training definition
class ScaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(512, 128), # Matches ResNet18
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.mlp(x)

def evaluate():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = "/home/junming/nobackup_junming/task4/training_data_dual"
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if len(all_files) == 0: return

    split = int(0.8 * len(all_files))
    train_files = all_files[:split]
    test_files  = all_files[split:]

    train_targets = [float(np.load(f)['target_scale']) for f in train_files]
    baseline = np.mean(train_targets)
    print(f"Baseline Guess: {baseline:.4f}")

    model = ScaleNet().to(DEVICE)
    model.load_state_dict(torch.load("scalenet_dual.pth", map_location=DEVICE))
    model.eval()

    err_base, err_ours = [], []

    for f in test_files:
        d = np.load(f)
        feat = torch.from_numpy(d['feature']).float().to(DEVICE)
        target = float(d['target_scale'])

        if len(feat.shape) == 1: feat = feat.unsqueeze(0)
        with torch.no_grad(): pred = model(feat).item()

        err_base.append(abs(baseline - target) / (target+1e-6))
        err_ours.append(abs(pred - target) / (target+1e-6))

    print("-" * 30)
    print(f"Base Error: {np.mean(err_base):.4f}")
    print(f"Ours Error: {np.mean(err_ours):.4f}")
    imp = ((np.mean(err_base)-np.mean(err_ours))/np.mean(err_base))*100
    print(f"Improvement: {imp:.2f}%")

if __name__ == "__main__":
    evaluate()