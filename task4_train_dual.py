import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import os
from torch.utils.data import Dataset, DataLoader

# --- Updated ScaleNet (Input 512) ---
class ScaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet18 outputs 512 features
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),  # <--- CHANGED from 1024
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.mlp(x)

class ScaleDataset(Dataset):
    def __init__(self):
        # Point to the NEW dual folder
        self.dir_path = "/home/junming/nobackup_junming/task4/training_data_dual"
        self.files = sorted(glob.glob(os.path.join(self.dir_path, "*.npz")))
        print(f"Loading {len(self.files)} samples from {self.dir_path}")

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        return torch.from_numpy(d['feature']).float().squeeze(), torch.tensor(d['target_scale']).float()

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = ScaleDataset()
    if len(dataset) == 0:
        print("No data found!")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = ScaleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("--- Training Dual-Stream ScaleNet ---")
    for epoch in range(25):
        total_loss = 0
        for feat, target in loader:
            feat, target = feat.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(feat)
            loss = nn.MSELoss()(pred.squeeze(), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "scalenet_dual.pth")
    print("Saved 'scalenet_dual.pth'")

if __name__ == "__main__":
    train()