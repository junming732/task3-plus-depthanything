from huggingface_hub import HfApi
import os

# --- CONFIGURATION ---
# Your User / Repo Name for the MODEL
REPO_ID = "Junming111/eth3d-scalenet-dual"

# Local Files to Upload
files_to_upload = {
    "scalenet_dual.pth": "scalenet_dual.pth",
    "task4_inference_dual.py": "inference.py"
}

# --- EXECUTION ---
# No login() needed if you ran 'huggingface-cli login' before
api = HfApi()

print(f"Ensuring repo exists: {REPO_ID}")
try:
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Repo check: {e}")

for local_path, remote_name in files_to_upload.items():
    if not os.path.exists(local_path):
        print(f"SKIPPING: {local_path} not found")
        continue

    print(f"Uploading {local_path} -> {remote_name}...")
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_name,
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("Done.")
    except Exception as e:
        print(f"Error uploading: {e}")

print("\nSUCCESS! Model files uploaded.")
print(f"Link: https://huggingface.co/{REPO_ID}")