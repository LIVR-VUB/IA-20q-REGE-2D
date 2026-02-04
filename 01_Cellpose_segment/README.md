# How to Download and Use REGE-Nusa 2D Cellpose Models

This guide provides comprehensive instructions for downloading the custom-trained Cellpose models hosted on the [LIVR-VUB/20q-REGE-2D](https://huggingface.co/LIVR-VUB/20q-REGE-2D) Hugging Face repository.

These models are optimized for:
*   **`cpsam_20q_cells`**: Whole-cell segmentation (cytoplasm).
*   **`cpsam_20qP_nuclei`**: Nuclei segmentation.

---

## Method 1: Automatic Download Script (Recommended)

We have provided a ready-to-use Python script that automatically handles the download and organizes the files for you.

### Step 1: Run the Script
Execute the following command in your terminal:

```bash
python3 download_models_20q.py
```

### What this script does
1.  Checks if you have the required `huggingface_hub` library.
2.  Connects to the `LIVR-VUB/20q-REGE-2D` repository.
3.  Downloads both model files.
4.  Saves them into a new folder named `downloaded_models` in the current directory.

---

## Method 2: Manual Download via Python

If you prefer to write your own code or integrate the download into an existing pipeline, use the following snippet.

### Prerequisites

Ensure you have the `huggingface_hub` library installed:

```bash
pip install huggingface_hub
```

### Code Snippet

```python
from huggingface_hub import hf_hub_download
import shutil
import os

repo_id = "LIVR-VUB/20q-REGE-2D"
filenames = ["cpsam_20q_cells", "cpsam_20qP_nuclei"]
output_dir = "./my_models"

os.makedirs(output_dir, exist_ok=True)

for f in filenames:
    print(f"Downloading {f}...")
    path = hf_hub_download(repo_id=repo_id, filename=f)
    # Move from cache to your local folder
    shutil.copy(path, os.path.join(output_dir, f))
    print(f"Saved to {os.path.join(output_dir, f)}")
```

---

## Method 3: Using Git (For Version Control)

If you are comfortable with Git, you can clone the entire repository. This is useful if you want to track changes to the models over time.

### Prerequisites
You must have `git` and `git-lfs` (Large File Storage) installed.

```bash
sudo apt-get install git git-lfs  # specialized for Ubuntu/Linux
git lfs install
```

### Command

```bash
git clone https://huggingface.co/LIVR-VUB/20q-REGE-2D
```

This will create a folder named `20q-REGE-2D` containing all models and documentation.

---

## How to Use the Models in Cellpose

Once downloaded, you can use these models with the standard Cellpose software, either via the Graphical User Interface (GUI) or the Python API.

### Option A: Using the GUI

1.  Open Cellpose: `cellpose`
2.  Drag and drop an image into the GUI.
3.  **Model Selection**: Instead of selecting a default model (like 'cyto'), go to **File > Load custom model**.
4.  Navigate to your download folder (e.g., `downloaded_models`) and select `cpsam_20q_cells` (for cells) or `cpsam_20qP_nuclei` (for nuclei).
5.  **Channel Settings**:
    *   **Cytoplasm**: Set the main channel to the cytoplasm stain.
    *   **Nuclei**: Set the main channel to DAPI/Nuclei stain.

### Option B: Using Python API

```python
from cellpose import models, io

# 1. Load the custom model
# Replace with the actual path to your downloaded model file
model_path = "./downloaded_models/cpsam_20q_cells" 
model = models.CellposeModel(gpu=True, pretrained_model=model_path)

# 2. Load your image
# image = io.imread('my_image.tif')

# 3. Run segmentation
# channels=[cytoplasm_channel, nucleus_channel] (e.g., [2, 3] or [1, 2])
# Set diameter=None to use the model's saved diameter size
masks, flows, styles = model.eval(image, diameter=None, channels=[1, 2])

print("Segmentation complete.")
```
