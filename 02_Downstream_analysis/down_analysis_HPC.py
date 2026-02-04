#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-folder quantification script.

Usage:
    python quantify.py --input_dir /path/to/input --output_dir /path/to/output

What this script does:
- Recursively reads all *_cX_img.tif and *_cX_cp.tif under input_dir.
- Processes each field-of-view.
- Mirrors the folder structure under output_dir.
- Writes one CSV per folder, named <FolderName>_per_cell_features.csv.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.measure import label, regionprops_table
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from scipy import ndimage as ndi

# -------------------------
# Marker definitions
# -------------------------
NUCLEAR_MARKERS = {
    "DAPI", "OCT4", "POU5F1",
    "PAX6", "TFAP2A", "EYA1", "SOX2",
}

CYTO_MARKERS = {
    "EPCAM", "KRT", "CK",
    "KRT8", "KRT18", "KRT5",
    "PMEL",
}

# -------------------------
# File grouping
# -------------------------

def group_files(input_dir: Path) -> Dict[str, Dict[str, Dict[int, Path]]]:
    """
    Group image and mask files by their base name.
    
    Expected filename pattern: <base>_c<channel_index>_<kind>.<ext>
    where <kind> is 'img' or 'cp', and <ext> is 'tif' or 'tiff'.
    
    Example: Demi_20q__c1-TFAP2A-647__c2-OCT4-594__c3-DAPI-405__idx-1_c1_img.tif
    """
    import re
    groups = {}

    # Search for both .tif and .tiff extensions
    from itertools import chain
    all_tifs = chain(input_dir.rglob("*.tif"), input_dir.rglob("*.tiff"))
    
    # Pattern to match _c<digit(s)>_<kind> at the end of the stem
    # This correctly handles cases like "_c1_img", "_c2_cp", or "_c1_img_cp"
    suffix_pattern = re.compile(r'^(.+)_c(\d+)_(img|img_cp|cp)$')
    
    for path in all_tifs:
        stem = path.stem
        
        match = suffix_pattern.match(stem)
        if not match:
            continue
        
        base = match.group(1)
        chan_idx = int(match.group(2))
        kind = match.group(3)

        g = groups.setdefault(base, {"imgs": {}, "masks": {}})
        if kind == "img":
            g["imgs"][chan_idx] = path
        elif kind in ("cp", "img_cp"):
            g["masks"][chan_idx] = path

    return groups

def parse_channel_mapping(base: str) -> Dict[int, str]:
    mapping = {}
    prefix = base.split("__idx")[0]
    segments = prefix.split("__")
    for seg in segments:
        if seg.startswith("c") and "-" in seg:
            ch = seg.split("-")[0][1:]
            marker = seg.split("-")[1].upper()
            try:
                idx = int(ch)
                mapping[idx] = marker
            except:
                pass
    return mapping

# -------------------------
# Image utilities
# -------------------------

def tifread(p: Path):
    return tiff.imread(str(p))

def load_img(p: Path):
    img = tifread(p)
    if img.ndim != 2:
        img = np.squeeze(img)
    return img.astype(np.float32)

def _ensure_label_mask(arr):
    arr = np.squeeze(arr)
    arr = arr.astype(np.int32)
    if arr.max() <= 1:
        arr = label(arr > 0)
    return arr

# -------------------------
# Mask selection
# -------------------------

def choose_nuclear_mask(masks, mapping):
    # Prefer DAPI
    for ch, p in masks.items():
        if mapping.get(ch, "").upper() == "DAPI":
            return _ensure_label_mask(tifread(p))

    # Otherwise nuclear markers
    for ch, p in masks.items():
        if mapping.get(ch, "").upper() in NUCLEAR_MARKERS:
            return _ensure_label_mask(tifread(p))

    return None

def choose_cell_mask(masks, mapping):
    for ch, p in masks.items():
        if mapping.get(ch, "").upper() in CYTO_MARKERS:
            return _ensure_label_mask(tifread(p))
    return None

# -------------------------
# Segmentation utilities
# -------------------------

def segment_nuc(nuc_img):
    try:
        th = threshold_otsu(nuc_img)
    except:
        th = nuc_img.mean() + nuc_img.std()

    bw = nuc_img > th
    bw = remove_small_objects(bw, 50)
    bw = ndi.binary_fill_holes(bw)

    dist = ndi.distance_transform_edt(bw)
    local_max = dist == ndi.maximum_filter(dist, size=5)
    markers = label(local_max)
    return watershed(-dist, markers, mask=bw).astype(np.int32)

def derive_cell_labels(nuc_labels, cyto_img):
    if cyto_img is not None:
        smoothed = gaussian(cyto_img, sigma=2)
        elevation = smoothed.max() - smoothed
        return watershed(elevation, markers=nuc_labels).astype(np.int32)
    else:
        dist = ndi.distance_transform_edt(nuc_labels > 0)
        return watershed(-dist, markers=nuc_labels).astype(np.int32)

# -------------------------
# Main per-group processing
# -------------------------

def process_group(base, group, input_root: Path):
    # Determine the folder this FOV came from
    if group["imgs"]:
        any_path = next(iter(group["imgs"].values()))
    elif group["masks"]:
        any_path = next(iter(group["masks"].values()))
    else:
        return []

    try:
        rel = any_path.parent.relative_to(input_root)
    except:
        rel = Path(".")

    rel_str = str(rel).replace("\\", "/")

    image_id = base
    cell_type = base.split("__")[0]
    genotype = "20q_mutant" if "20q" in base.lower() else "WT"

    mapping = parse_channel_mapping(base)

    # Load images
    channels = {}
    for ch_idx, p in group["imgs"].items():
        marker = mapping.get(ch_idx)
        if marker:
            channels[marker] = load_img(p)

    if not channels:
        return []

    nuc_mask = choose_nuclear_mask(group["masks"], mapping)
    cell_mask = choose_cell_mask(group["masks"], mapping)

    if nuc_mask is None:
        # Auto segment nuclei
        nuc_markers = [m for m in channels.keys() if m.upper() in NUCLEAR_MARKERS]
        if not nuc_markers:
            return []
        nuc_mask = segment_nuc(channels[nuc_markers[0]])

    if cell_mask is None:
        cyto_markers = [m for m in channels.keys() if m.upper() in CYTO_MARKERS]
        cyto_img = channels[cyto_markers[0]] if cyto_markers else None
        cell_mask = derive_cell_labels(nuc_mask, cyto_img)

    # Measure nuclei
    nuc_props = regionprops_table(nuc_mask, properties=("label", "area"))
    nuc_df = pd.DataFrame(nuc_props)
    nuc_df.rename(columns={"label": "CellID", "area": "NucleusArea"}, inplace=True)

    rows = []

    for _, row in nuc_df.iterrows():
        cell_id = int(row["CellID"])
        if cell_id == 0:
            continue

        nuc_region = nuc_mask == cell_id
        cell_region = cell_mask == cell_id

        if not np.any(cell_region):
            continue

        info = {
            "ImageID": image_id,
            "CellType": cell_type,
            "Genotype": genotype,
            "CellID": cell_id,
            "NucleusArea": float(row["NucleusArea"]),
            "SourceRelDir": rel_str,
        }

        for marker, img_ch in channels.items():
            m = marker.upper()

            region = nuc_region if m in NUCLEAR_MARKERS else cell_region

            if np.any(region):
                pixels = img_ch[region]
                info[f"{m}_int"] = float(pixels.sum())
                info[f"{m}_mean"] = float(pixels.mean())
            else:
                info[f"{m}_int"] = 0.0
                info[f"{m}_mean"] = 0.0

        rows.append(info)

    return rows

# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quantify per-cell features and write per-folder CSVs."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    groups = group_files(input_root)

    all_rows = []

    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {
            ex.submit(process_group, base, group, input_root): base
            for base, group in groups.items()
        }

        for fut in as_completed(futures):
            try:
                rows = fut.result()
                all_rows.extend(rows)
            except Exception as e:
                print(f"[WARN] Error: {e}")

    if not all_rows:
        print("[ERROR] No cells quantified.")
        return

    df = pd.DataFrame(all_rows)

    # Write per-folder CSVs
    for rel_dir, sub_df in df.groupby("SourceRelDir"):
        out_folder = output_root / Path(rel_dir)
        out_folder.mkdir(parents=True, exist_ok=True)

        folder_name = out_folder.name or "root"
        csv_path = out_folder / f"{folder_name}_per_cell_features.csv"

        sub_df.to_csv(csv_path, index=False)
        print(f"[OK] Wrote {len(sub_df)} rows to {csv_path}")


if __name__ == "__main__":
    main()
