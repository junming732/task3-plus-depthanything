import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import os
from torch.utils.data import Dataset, DataLoader

class ScaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1024 features. Output: 1 Scalar (The Scale Factor)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.mlp(x)

class ScaleDataset(Dataset):
    def __init__(self, dir_path):
        self.files = glob.glob(os.path.join(dir_path, "*.npz"))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        return torch.from_numpy(d['feature']).float().squeeze(), torch.tensor(d['target_scale']).float()

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = ScaleDataset("task4_training_data")
    if len(dataset) == 0:
        print("No data. Run prepare script.")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = ScaleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("--- Training ScaleNet (Scene Level) ---")
    for epoch in range(20):
        total_loss = 0
        for feat, target in loader:
            feat, target = feat.to(device), target.to(device)
            optimizer.zero_grad()
            pred = model(feat)
            loss = nn.MSELoss()(pred.squeeze(), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Loss {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "scalenet_best.pth")
    print("Saved.")

if __name__ == "__main__":
    train()