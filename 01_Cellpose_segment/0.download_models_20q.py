import os
import sys
import shutil

def install_package(package):
    """Helper to install a package if missing."""
    print(f"Installing {package}...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check for huggingface_hub
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("The 'huggingface_hub' library is required but not installed.")
    install_package("huggingface_hub")
    from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "LIVR-VUB/20q-REGE-2D"
DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_models")
MODELS = {
    "cpsam_20q_cells": "Cell Segmentation Model (Cytoplasm)",
    "cpsam_20qP_nuclei": "Nuclei Segmentation Model"
}

def download_models():
    """Downloads models from Hugging Face Hub."""
    print("===================================================")
    print(f"Downloading models from: https://huggingface.co/{REPO_ID}")
    print(f"Destination folder: {DOWNLOAD_DIR}")
    print("===================================================\n")

    # Create destination directory if it doesn't exist
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for filename, description in MODELS.items():
        print(f"Downloading {filename} ({description})...")
        try:
            # Download the file
            # cache_dir is optional; if omitted, it uses the default HF cache.
            # We copy it to a specific local folder for easier access.
            cached_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
            
            # Copy from cache to our target directory
            target_path = os.path.join(DOWNLOAD_DIR, filename)
            shutil.copy2(cached_path, target_path)
            
            print(f" [OK] Processed: {target_path}")
        except Exception as e:
            print(f" [ERROR] Failed to download {filename}: {e}")
            print("  Please check your internet connection or if the model exists in the repo.")

    print("\n===================================================")
    print("Download process finished.")
    print(f"Your models are located in: {DOWNLOAD_DIR}")
    print("===================================================")

if __name__ == "__main__":
    download_models()
