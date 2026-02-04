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

Now with:
   - Publication-style Matplotlib theme
   - Colorblind-safe palettes (Paul Tol + RColorBrewer Pastel1)
   - Tall, figure-panel-friendly stacked bar plots
   - PNG + PDF + SVG export at high DPI
   - Recursive search + per-dataset output (FULL relative folder preserved)
"""

import os
import re
import argparse
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# =============================================================================
#  Publication-style theme and helpers
# =============================================================================

_NEW_BLACK = "#373737"

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.labelpad": 2,
    "axes.titlepad": 4,
    "axes.linewidth": 0.5,
    "lines.linewidth": 0.5,
    "xtick.major.size": 2,
    "xtick.major.pad": 1,
    "xtick.major.width": 0.5,
    "ytick.major.size": 2,
    "ytick.major.pad": 1,
    "ytick.major.width": 0.5,
    "xtick.minor.size": 2,
    "xtick.minor.width": 0.5,
    "ytick.minor.size": 2,
    "ytick.minor.width": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "text.color": _NEW_BLACK,
    "patch.edgecolor": _NEW_BLACK,
    "patch.force_edgecolor": False,
    "hatch.color": _NEW_BLACK,
    "axes.edgecolor": _NEW_BLACK,
    "axes.labelcolor": _NEW_BLACK,
    "xtick.color": _NEW_BLACK,
    "ytick.color": _NEW_BLACK,
    "figure.figsize": (2.2, 3.4),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.dpi": 600,
}

if sns is not None:
    sns.set_theme(style="ticks", rc=_RC)
else:
    mpl.rcParams.update(_RC)

EXPORT_DPI = 600


def tol_palette(name: str = "vibrant"):
    """Paul Tol colorblind-safe categorical palettes."""
    sets = {
        "bright": [
            "#4477AA", "#EE6677", "#228833", "#CCBB44",
            "#66CCEE", "#AA3377", "#BBBBBB", "#000000"
        ],
        "high-contrast": ["#004488", "#DDAA33", "#BB5566", "#000000"],
        "vibrant": [
            "#EE7733", "#0077BB", "#33BBEE", "#EE3377",
            "#CC3311", "#009988", "#BBBBBB", "#000000"
        ],
        "muted": [
            "#CC6677", "#332288", "#DDCC77", "#117733",
            "#88CCEE", "#882255", "#44AA99", "#999933",
            "#AA4499", "#DDDDDD", "#000000"
        ],
        "medium-contrast": [
            "#6699CC", "#004488", "#EECC66", "#994455",
            "#997700", "#EE99AA", "#000000"
        ],
        "light": [
            "#77AADD", "#EE8866", "#EEDD88", "#FFAABB",
            "#99DDFF", "#44BB99", "#BBCC33", "#AAAA00",
            "#DDDDDD", "#000000"
        ],
    }
    return sets.get(name, sets["vibrant"])


def pastel1_palette():
    """
    RColorBrewer 'Pastel1' palette (9 colors).
    Source: RColorBrewer::brewer.pal(9, "Pastel1")
    """
    return [
        "#FBB4AE",
        "#B3CDE3",
        "#CCEBC5",
        "#DECBE4",
        "#FED9A6",
        "#FFFFCC",
        "#E5D8BD",
        "#FDDAEC",
        "#F2F2F2",
    ]


def tidy_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def save_figure(fig, output_path: str, dpi: int = EXPORT_DPI):
    base, ext = os.path.splitext(output_path)
    if ext == "":
        output_path = base + ".png"
        base = output_path[:-4]

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    for extra_ext in [".pdf", ".svg"]:
        fig.savefig(base + extra_ext, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# =============================================================================
#  Core logic
# =============================================================================

def parse_filename_metadata(filepath):
    filename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(filename)[0]

    sample_match = re.match(r"^([^_]+_[^_]+)", name_no_ext)
    sample_name = sample_match.group(1) if sample_match else "Unknown"

    idx_match = re.search(r"__idx-(\d+)", name_no_ext)
    idx = int(idx_match.group(1)) if idx_match else 0

    marker_map = {}
    matches = re.finditer(r"c(\d+)-([A-Za-z0-9]+)-", name_no_ext)
    for m in matches:
        ch = int(m.group(1))
        marker_name = m.group(2)
        marker_map[ch] = marker_name

    # Match both _c<digit>_cp and _c<digit>_img_cp patterns
    ch_file_match = re.search(r"_c(\d+)(?:_img)?_cp", name_no_ext)
    file_channel = int(ch_file_match.group(1)) if ch_file_match else None

    return {
        "filepath": filepath,
        "filename": filename,
        "sample_name": sample_name,
        "idx": idx,
        "marker_map": marker_map,
        "file_channel": file_channel,
    }


def group_files_by_image_set(input_dir):
    """
    EXACT recursive strategy inspired by script-1:

    - Recursively scan *all* subfolders under input_dir
    - Dataset key = FULL relative parent directory of each file:
          dataset = str(f.parent.relative_to(input_path))
      (like script-1's RelativePath logic)

    Outputs will therefore mirror the full input tree, e.g.:
        input_dir/A/B/C/file_cp.tif
        -> output_dir/A/B/C/Analysis_<combo>/...
    """
    input_path = Path(input_dir)

    # Robust recursive scan for *_cp.tif / *_cp.tiff (case-insensitive)
    files = []
    for p in input_path.rglob("*"):
        if not p.is_file():
            continue
        name_l = p.name.lower()
        if name_l.endswith("_cp.tif") or name_l.endswith("_cp.tiff"):
            files.append(p)

    # key = (dataset, sample_name, idx, marker_signature)
    image_sets = defaultdict(dict)

    for f in files:
        meta = parse_filename_metadata(str(f))
        if meta["file_channel"] is None or not meta["marker_map"]:
            continue

        # FULL relative parent folder (script-1 style)
        try:
            rel_parent = f.parent.relative_to(input_path)
        except Exception:
            rel_parent = Path(".")

        dataset = str(rel_parent).replace("\\", "/") if str(rel_parent) else "."

        marker_signature = tuple(sorted(meta["marker_map"].items()))
        key = (dataset, meta["sample_name"], meta["idx"], marker_signature)
        image_sets[key][meta["file_channel"]] = str(f)

    return image_sets


def classify_image_set(image_set_paths, marker_map, overlap_threshold=0.15):
    dapi_ch = None
    for ch, name in marker_map.items():
        if "DAPI" in name.upper():
            dapi_ch = ch
            break

    if dapi_ch is None or dapi_ch not in image_set_paths:
        dapi_ch = max(image_set_paths.keys())

    dapi_mask = imread(image_set_paths[dapi_ch])
    if dapi_mask.ndim > 2:
        dapi_mask = dapi_mask.squeeze()

    cell_ids = np.unique(dapi_mask)
    cell_ids = cell_ids[cell_ids != 0]

    results = []

    other_markers = {}
    for ch, path in image_set_paths.items():
        if ch == dapi_ch:
            continue
        marker_name = marker_map[ch]
        mask = imread(path)
        if mask.ndim > 2:
            mask = mask.squeeze()
        other_markers[marker_name] = mask

    sorted_marker_names = sorted(other_markers.keys())

    for cid in cell_ids:
        nucleus_bool = dapi_mask == cid
        nuc_area = np.sum(nucleus_bool)

        row = {"cell_id": cid, "nucleus_area": nuc_area}
        status_parts = []

        coords = np.where(nucleus_bool)
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        nuc_crop = nucleus_bool[y_min:y_max + 1, x_min:x_max + 1]

        for m_name in sorted_marker_names:
            m_mask = other_markers[m_name]
            marker_crop = m_mask[y_min:y_max + 1, x_min:x_max + 1]
            intersection = np.logical_and(nuc_crop, marker_crop > 0).sum()
            overlap_frac = intersection / float(nuc_area)

            is_positive = overlap_frac >= overlap_threshold
            row[m_name] = is_positive

            sign = "+" if is_positive else "-"
            status_parts.append(f"{m_name} {sign}")

        row["category"] = " ".join(status_parts) if status_parts else "DAPI only"
        results.append(row)

    return results


def create_stacked_bar_plot(df, x_col, title, output_path, category_colors=None):
    """
    Tall, publication-style stacked bar plot using RColorBrewer 'Pastel1'.
    """
    counts = df.groupby([x_col, "category"]).size().reset_index(name="count")
    totals = df.groupby([x_col]).size().reset_index(name="total")
    data = pd.merge(counts, totals, on=x_col)
    data["fraction"] = data["count"] / data["total"]

    pivot_df = (
        data.pivot(index=x_col, columns="category", values="fraction").fillna(0.0)
    )
    categories = list(pivot_df.columns)

    if category_colors is None:
        palette = pastel1_palette()
        category_colors = {
            cat: palette[i % len(palette)] for i, cat in enumerate(categories)
        }

    n_bars = len(pivot_df.index)
    width = max(2.0, min(4.0, 0.5 * n_bars))
    height = 3.4
    fig, ax = plt.subplots(
        figsize=(width, height),
        dpi=EXPORT_DPI,
        constrained_layout=True
    )

    x_labels = pivot_df.index.tolist()
    x_pos = np.arange(len(x_labels))

    bottom = np.zeros(len(x_labels))
    for cat in categories:
        vals = pivot_df[cat].values
        ax.bar(
            x_pos,
            vals,
            bottom=bottom,
            color=category_colors.get(cat, "#BBBBBB"),
            edgecolor="none",
            width=0.65,
            label=cat,
        )
        bottom += vals

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Fraction of population")
    ax.set_title(title, pad=6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.4)
    ax.grid(False, axis="x")

    ax.legend(
        title="Phenotype",
        frameon=False,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
    )
    tidy_axes(ax)

    save_figure(fig, output_path, dpi=EXPORT_DPI)

    return category_colors


# =============================================================================
#  Parallel helper
# =============================================================================

def _process_image_set_task(args):
    dataset, sample, idx, marker_sig, image_set_paths, overlap_threshold = args

    marker_map = dict(marker_sig)

    markers_only = [v for _, v in marker_map.items() if "DAPI" not in v.upper()]
    combo_name = "_".join(sorted(markers_only)) if markers_only else "DAPI_only"

    print(f"   Processing [{dataset}] {sample} idx-{idx} [{combo_name}]...")

    results = classify_image_set(image_set_paths, marker_map, overlap_threshold)
    if not results:
        return None

    df = pd.DataFrame(results)
    df["dataset"] = dataset
    df["sample"] = sample
    df["idx"] = idx
    df["combination"] = combo_name
    df["full_id"] = f"{sample}_idx-{idx}"

    return dataset, combo_name, df


def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect markers and classify cell populations."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Root folder containing *_cp.tif files under subfolders.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Folder to save results"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.15, help="Overlap threshold (0-1)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of parallel workers (use 1 for serial processing)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("1. Scanning files and grouping by experiment...")
    image_sets = group_files_by_image_set(args.input_dir)

    if not image_sets:
        print("No valid file sets found. Check input directory.")
        return

    print(f"   Found {len(image_sets)} individual image sets.")

    experiments = defaultdict(lambda: defaultdict(list))

    task_params = []
    for (dataset, sample, idx, marker_sig), paths in image_sets.items():
        task_params.append((dataset, sample, idx, marker_sig, paths, args.overlap))

    if args.n_workers <= 1:
        for params in task_params:
            res = _process_image_set_task(params)
            if res is None:
                continue
            dataset, combo_name, df = res
            experiments[dataset][combo_name].append(df)
    else:
        print(f"   Using {args.n_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            future_to_params = {
                pool.submit(_process_image_set_task, params): params
                for params in task_params
            }
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                dataset, sample, idx, marker_sig, paths, overlap = params
                try:
                    res = future.result()
                except Exception as exc:
                    print(f"   Error processing [{dataset}] {sample} idx-{idx}: {exc}")
                    continue
                if res is None:
                    continue
                dataset_result, combo_name, df = res
                experiments[dataset_result][combo_name].append(df)

    print("\n2. Generating plots...")

    for dataset, combo_dict in experiments.items():
        for combo_name, df_list in combo_dict.items():
            if not df_list:
                continue

            combined_df = pd.concat(df_list, ignore_index=True)

            # dataset is now FULL relative folder path (script-1 style)
            dataset_dir = args.output_dir if dataset in (".", "", None) else os.path.join(args.output_dir, dataset)

            combo_dir = os.path.join(dataset_dir, f"Analysis_{combo_name}")
            os.makedirs(combo_dir, exist_ok=True)

            csv_path = os.path.join(combo_dir, "cell_data.csv")
            combined_df.to_csv(csv_path, index=False)

            unique_cats = sorted(combined_df["category"].unique())
            pastel = pastel1_palette()
            color_map = {cat: pastel[i % len(pastel)] for i, cat in enumerate(unique_cats)}

            print(f"   Creating aggregated plot for [{dataset}] {combo_name}...")
            agg_plot_path = os.path.join(combo_dir, f"Aggregated_{combo_name}_Distribution.png")
            create_stacked_bar_plot(
                combined_df,
                x_col="sample",
                title=f"Population distribution: {combo_name}\n(aggregated replicates; {dataset})",
                output_path=agg_plot_path,
                category_colors=color_map,
            )

            print(f"   Creating individual plots for [{dataset}] {combo_name}...")
            indiv_dir = os.path.join(combo_dir, "Individual_Plots")
            os.makedirs(indiv_dir, exist_ok=True)

            indiv_plot_path = os.path.join(indiv_dir, f"Individual_{combo_name}_Distribution.png")
            create_stacked_bar_plot(
                combined_df,
                x_col="full_id",
                title=f"Population distribution: {combo_name}\n(individual images; {dataset})",
                output_path=indiv_plot_path,
                category_colors=color_map,
            )

    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
