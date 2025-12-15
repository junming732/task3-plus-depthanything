from huggingface_hub import HfApi
import os
import shutil

# --- CONFIGURATION ---
REPO_ID = "Junming111/eth3d-scalenet-dual"

# 1. Define the specific images you want to upload
# We take one "Courtyard" (Outdoor) and one "Pipes" (Indoor/Industrial)
images_to_upload = {
    # Local Path on Server : Name on Hugging Face
    "/home/junming/nobackup_junming/eth3d-dataset/train/undistorted/courtyard/images/dslr_images_undistorted/DSC_0286.JPG": "samples/test_courtyard.jpg",
    "/home/junming/nobackup_junming/eth3d-dataset/train/undistorted/pipes/images/dslr_images_undistorted/DSC_0219.JPG":     "samples/test_pipes.jpg"
}

# --- EXECUTION ---
api = HfApi()

print(f"Targeting Repo: {REPO_ID}")

for local_path, remote_path in images_to_upload.items():
    if not os.path.exists(local_path):
        print(f"SKIPPING: Could not find {local_path}")
        continue

    print(f"Uploading {local_path} -> {remote_path}...")
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=REPO_ID,
            repo_type="model" # We put images inside the model repo for easy access
        )
        print("Done.")
    except Exception as e:
        print(f"Error uploading: {e}")

print("\nSUCCESS! Sample images live at:")
print(f"https://huggingface.co/{REPO_ID}/tree/main/samples")