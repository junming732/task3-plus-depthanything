from huggingface_hub import HfApi
import os
import shutil

# --- CONFIGURATION ---
REPO_ID = "Junming111/eth3d-task4-data"
SOURCE_FOLDER = "/home/junming/nobackup_junming/task4/training_data_dual"
ZIP_NAME = "task4_training_data_dual.zip"

# --- 1. ZIP ---
print(f"Zipping folder: {SOURCE_FOLDER}...")
shutil.make_archive("task4_data_temp", 'zip', SOURCE_FOLDER)
local_zip_path = "task4_data_temp.zip"

# --- 2. UPLOAD ---
api = HfApi()

print(f"Ensuring repo exists: {REPO_ID}")
try:
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
except Exception as e:
    print(f"Repo check: {e}")

print(f"Uploading {local_zip_path}...")
try:
    api.upload_file(
        path_or_fileobj=local_zip_path,
        path_in_repo=ZIP_NAME,
        repo_id=REPO_ID,
        repo_type="dataset"
    )
    print("\nSUCCESS! Dataset uploaded.")
    print(f"Link: https://huggingface.co/datasets/{REPO_ID}/blob/main/{ZIP_NAME}")
except Exception as e:
    print(f"Upload failed: {e}")