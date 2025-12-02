#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dynamic_marker_classification.py
================================

1. Detects marker combinations automatically from filenames.
2. Groups images by marker sets (e.g., "EYA1+PAX6" vs "OCT4+EPCAM").
3. Classifies nuclei based on Cellpose mask overlaps.
4. Generates:
   - Main Stacked Bar Plot: Aggregates all indices (idx-1, idx-2...) into one bar per sample.
   - Individual Plots: Saves separate plots for every image in a subfolder.

Usage:
    python dynamic_marker_classification.py --input_dir /path/to/files --output_dir /path/to/results
"""

import os
import re
import glob
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def parse_filename_metadata(filepath):
    """
    Parses complex filenames to extract sample info and marker map.
    
    Expected format examples:
    Nemo_20q__c1-EYA1-647__c2-PAX6-594__c3-DAPI-405__idx-1_c3_cp.tif
    """
    filename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(filename)[0]

    # 1. Extract Sample Name (everything before the first double underscore)
    sample_match = re.match(r"^([^_]+_[^_]+)", name_no_ext)
    sample_name = sample_match.group(1) if sample_match else "Unknown"

    # 2. Extract Index (idx-N)
    idx_match = re.search(r"__idx-(\d+)", name_no_ext)
    idx = int(idx_match.group(1)) if idx_match else 0

    # 3. Extract Channel-Marker mapping
    # Looks for patterns like "c1-EYA1-647"
    marker_map = {}
    # Find all matches of c(number)-(MarkerName)-
    # We ignore the wavelength (number at the end)
    matches = re.finditer(r"c(\d+)-([A-Za-z0-9]+)-", name_no_ext)
    for m in matches:
        ch = int(m.group(1))
        marker_name = m.group(2)
        marker_map[ch] = marker_name

    # 4. Identify which channel this specific file represents
    # Ends with _cN_cp.tif or _cN_img.tif
    ch_file_match = re.search(r"_c(\d+)_cp", name_no_ext)
    file_channel = int(ch_file_match.group(1)) if ch_file_match else None

    return {
        "filepath": filepath,
        "filename": filename,
        "sample_name": sample_name,
        "idx": idx,
        "marker_map": marker_map,
        "file_channel": file_channel
    }

def group_files_by_image_set(input_dir):
    """
    Scans directory and groups files that belong to the same image (same sample + same idx).
    """
    files = glob.glob(os.path.join(input_dir, "*_cp.tif")) + glob.glob(os.path.join(input_dir, "*_cp.tiff"))
    
    # Key: (sample_name, idx, marker_signature_tuple)
    # Value: dict of {channel: filepath}
    image_sets = defaultdict(dict)
    
    for f in files:
        meta = parse_filename_metadata(f)
        if meta['file_channel'] is None or not meta['marker_map']:
            continue
            
        # Create a unique signature for the marker combination to handle different experiments
        # e.g., tuple(("c1", "EYA1"), ("c2", "PAX6"), ("c3", "DAPI"))
        marker_signature = tuple(sorted(meta['marker_map'].items()))
        
        key = (meta['sample_name'], meta['idx'], marker_signature)
        image_sets[key][meta['file_channel']] = f

    return image_sets

def classify_image_set(image_set_paths, marker_map, overlap_threshold=0.15):
    """
    Loads masks and classifies cells.
    Strategy:
    1. Identify DAPI channel. Use DAPI masks as the "cells".
    2. Check overlap of DAPI masks with masks from other channels.
    """
    # 1. Find DAPI channel (Nuclear Anchor)
    dapi_ch = None
    for ch, name in marker_map.items():
        if "DAPI" in name.upper():
            dapi_ch = ch
            break
    
    if dapi_ch is None or dapi_ch not in image_set_paths:
        # Fallback: Use the highest channel number if DAPI is missing/not named
        dapi_ch = max(image_set_paths.keys())
    
    # Load DAPI Mask (The Base)
    dapi_mask = imread(image_set_paths[dapi_ch])
    if dapi_mask.ndim > 2: dapi_mask = dapi_mask.squeeze() # Handle stacks if necessary
    
    cell_ids = np.unique(dapi_mask)
    cell_ids = cell_ids[cell_ids != 0] # Remove background
    
    results = []
    
    # Load other markers
    other_markers = {}
    for ch, path in image_set_paths.items():
        if ch == dapi_ch: continue
        
        marker_name = marker_map[ch]
        mask = imread(path)
        if mask.ndim > 2: mask = mask.squeeze()
        other_markers[marker_name] = mask

    sorted_marker_names = sorted(other_markers.keys())

    # Analyze every cell
    for cid in cell_ids:
        # Boolean mask for current nucleus
        nucleus_bool = (dapi_mask == cid)
        nuc_area = np.sum(nucleus_bool)
        
        row = {"cell_id": cid, "nucleus_area": nuc_area}
        
        status_parts = []
        
        for m_name in sorted_marker_names:
            m_mask = other_markers[m_name]
            
            # Fast overlap check: 
            # Get the region of interest to speed up (slicing)
            coords = np.where(nucleus_bool)
            y_min, y_max = np.min(coords[0]), np.max(coords[0])
            x_min, x_max = np.min(coords[1]), np.max(coords[1])
            
            nuc_crop = nucleus_bool[y_min:y_max+1, x_min:x_max+1]
            marker_crop = m_mask[y_min:y_max+1, x_min:x_max+1]
            
            # Intersection: Nucleus exists AND Marker mask > 0
            intersection = np.logical_and(nuc_crop, marker_crop > 0).sum()
            overlap_frac = intersection / nuc_area
            
            is_positive = overlap_frac >= overlap_threshold
            row[m_name] = is_positive
            
            sign = "+" if is_positive else "-"
            status_parts.append(f"{m_name}{sign}")
            
        row["category"] = " ".join(status_parts) if status_parts else "DAPI Only"
        results.append(row)
        
    return results

def create_stacked_bar_plot(df, x_col, title, output_path, category_colors=None):
    """
    Generic plotting function.
    df: DataFrame containing ['category', x_col]
    """
    # Calculate fractions
    counts = df.groupby([x_col, 'category']).size().reset_index(name='count')
    totals = df.groupby([x_col]).size().reset_index(name='total')
    data = pd.merge(counts, totals, on=x_col)
    data['fraction'] = data['count'] / data['total']
    
    # Pivot for plotting
    pivot_df = data.pivot(index=x_col, columns='category', values='fraction').fillna(0)
    
    # Ensure consistent colors
    categories = sorted(pivot_df.columns)
    if category_colors is None:
        # Generate colors if not provided
        palette = sns.color_palette("husl", len(categories))
        category_colors = dict(zip(categories, palette))
    
    # Plot
    ax = pivot_df.plot(kind='bar', stacked=True, figsize=(10, 7), 
                       color=[category_colors.get(x, '#333333') for x in categories],
                       width=0.85)
    
    plt.title(title, fontsize=14)
    plt.ylabel("Fraction of Population")
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Phenotype")
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return category_colors

def main():
    parser = argparse.ArgumentParser(description="Auto-detect markers and classify cell populations.")
    parser.add_argument("--input_dir", required=True, help="Folder containing _cp.tif files")
    parser.add_argument("--output_dir", required=True, help="Folder to save results")
    parser.add_argument("--overlap", type=float, default=0.15, help="Overlap threshold (0-1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("1. Scanning files and grouping by experiment...")
    image_sets = group_files_by_image_set(args.input_dir)
    
    if not image_sets:
        print("No valid file sets found. Check input directory.")
        return

    # Organize data by Marker Combination
    # Key: Tuple of markers (e.g., "EPCAM_OCT4"), Value: List of DataFrames
    experiments = defaultdict(list)

    print(f"   Found {len(image_sets)} individual image sets.")

    # Process each image
    for (sample, idx, marker_sig), paths in image_sets.items():
        # Convert marker signature tuple back to dict for classification
        marker_map = dict(marker_sig)
        
        # Create a readable string for this combination, e.g., "EPCAM_OCT4"
        # We exclude DAPI from the combination name to keep it clean
        markers_only = [v for k,v in marker_map.items() if "DAPI" not in v.upper()]
        combo_name = "_".join(sorted(markers_only))
        
        print(f"   Processing {sample} idx-{idx} [{combo_name}]...")
        
        results = classify_image_set(paths, marker_map, args.overlap)
        
        if not results:
            continue
            
        df = pd.DataFrame(results)
        df['sample'] = sample
        df['idx'] = idx
        df['combination'] = combo_name
        df['full_id'] = f"{sample}_idx-{idx}"
        
        experiments[combo_name].append(df)

    # Visualization Phase
    print("\n2. Generating Plots...")
    
    for combo_name, df_list in experiments.items():
        if not df_list: continue
        
        # Merge all data for this specific marker combination
        combined_df = pd.concat(df_list, ignore_index=True)
        
        # Define output folder for this combination
        combo_dir = os.path.join(args.output_dir, f"Analysis_{combo_name}")
        os.makedirs(combo_dir, exist_ok=True)
        
        # CSV Export
        csv_path = os.path.join(combo_dir, "cell_data.csv")
        combined_df.to_csv(csv_path, index=False)
        
        # --- PLOT 1: Aggregated (Main Request) ---
        # Group by 'sample' only (combining idx 1, 2, 3)
        print(f"   Creating aggregated plot for {combo_name}...")
        agg_plot_path = os.path.join(combo_dir, f"Aggregated_{combo_name}_Distribution.png")
        
        # We calculate consistent colors based on all available categories in this combo
        unique_cats = sorted(combined_df['category'].unique())
        palette = sns.color_palette("husl", len(unique_cats))
        color_map = dict(zip(unique_cats, palette))
        
        create_stacked_bar_plot(
            combined_df, 
            x_col='sample', 
            title=f"Population Distribution: {combo_name}\n(Aggregated Replicates)", 
            output_path=agg_plot_path,
            category_colors=color_map
        )
        
        # --- PLOT 2: Individual Files (Separate Subfolder) ---
        print(f"   Creating individual plots for {combo_name}...")
        indiv_dir = os.path.join(combo_dir, "Individual_Plots")
        os.makedirs(indiv_dir, exist_ok=True)
        
        # We plot using 'full_id' which is "Sample_idx-N"
        indiv_plot_path = os.path.join(indiv_dir, f"Individual_{combo_name}_Distribution.png")
        create_stacked_bar_plot(
            combined_df,
            x_col='full_id',
            title=f"Population Distribution: {combo_name}\n(Individual Images)",
            output_path=indiv_plot_path,
            category_colors=color_map # Reuse same colors
        )

    print(f"\nDone! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
