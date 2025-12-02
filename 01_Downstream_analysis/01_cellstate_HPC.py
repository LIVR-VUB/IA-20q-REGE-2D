#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel batch version of simple_analysis.py

- Recursively finds all *_per_cell_features.csv under --root_dir
- For each CSV, runs the same analyses as simple_analysis.py:
    * summary statistics
    * fraction positive
    * morphology statistics
    * correlation matrix
    * coexpression table for OCT4/POU5F1 vs PAX6 (if present)
    * histograms (if plots enabled)
    * scatter plot for OCT4/POU5F1 vs PAX6 (if present & plots enabled)
- All outputs are written next to each CSV (same folder).
"""

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # important for HPC
import matplotlib.pyplot as plt

import argparse
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from tqdm import tqdm

# Limit BLAS/OpenMP threading inside each process
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# -------------------------------------------------------------------
# Core analysis functions (copied from simple_analysis.py)
# -------------------------------------------------------------------

def summary_statistics(df: pd.DataFrame, markers: List[str]) -> pd.DataFrame:
    rows = []
    for marker in markers:
        mean_col = f"{marker.upper()}_mean"
        if mean_col not in df.columns:
            continue
        for genotype in df["Genotype"].unique():
            sub = df[df["Genotype"] == genotype]
            vals = sub[mean_col].dropna()
            if len(vals) == 0:
                continue
            rows.append(
                {
                    "Genotype": genotype,
                    "Marker": marker.upper(),
                    "n_cells": int(len(vals)),
                    "mean_intensity": float(np.mean(vals)),
                    "median_intensity": float(np.median(vals)),
                    "std_intensity": float(np.std(vals, ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def _compute_threshold(values: np.ndarray) -> float:
    try:
        thresh = threshold_otsu(values)
    except Exception:
        thresh = float(np.mean(values) + 3 * np.std(values))
    return float(thresh)


def fraction_positive(df: pd.DataFrame, markers: List[str]) -> pd.DataFrame:
    results = []
    for marker in markers:
        mean_col = f"{marker.upper()}_mean"
        if mean_col not in df.columns:
            continue
        all_vals = df[mean_col].dropna().values
        if len(all_vals) == 0:
            continue
        thresh = _compute_threshold(all_vals)
        for genotype in df["Genotype"].unique():
            sub = df[df["Genotype"] == genotype]
            vals = sub[mean_col].dropna()
            if len(vals) == 0:
                continue
            pos_frac = float((vals > thresh).mean()) * 100.0
            results.append(
                {
                    "Genotype": genotype,
                    "Marker": marker.upper(),
                    "threshold": float(thresh),
                    "percent_positive": pos_frac,
                    "n_cells": int(len(vals)),
                }
            )
    return pd.DataFrame(results)


def coexpression_table(df: pd.DataFrame, marker1: str, marker2: str) -> pd.DataFrame:
    m1_col = f"{marker1.upper()}_mean"
    m2_col = f"{marker2.upper()}_mean"
    if m1_col not in df.columns or m2_col not in df.columns:
        raise ValueError("Specified markers not found in DataFrame")
    m1_thresh = _compute_threshold(df[m1_col].dropna().values)
    m2_thresh = _compute_threshold(df[m2_col].dropna().values)
    table_rows = []
    for genotype in df["Genotype"].unique():
        sub = df[df["Genotype"] == genotype]
        m1_vals = sub[m1_col]
        m2_vals = sub[m2_col]
        pos1 = m1_vals > m1_thresh
        pos2 = m2_vals > m2_thresh
        both_pos = int((pos1 & pos2).sum())
        only1 = int((pos1 & ~pos2).sum())
        only2 = int((~pos1 & pos2).sum())
        neither = int((~pos1 & ~pos2).sum())
        total = int(len(sub))
        for category, count in [
            ("both_positive", both_pos),
            (f"{marker1.upper()}_only", only1),
            (f"{marker2.upper()}_only", only2),
            ("double_negative", neither),
        ]:
            table_rows.append(
                {
                    "Genotype": genotype,
                    "Category": category,
                    "count": int(count),
                    "percent": (count / total * 100.0) if total > 0 else np.nan,
                }
            )
    return pd.DataFrame(table_rows)


def morph_statistics(df: pd.DataFrame) -> pd.DataFrame:
    morph_cols = []
    for col in df.columns:
        if col.endswith("_int") or col.endswith("_mean"):
            continue
        if col in ("ImageID", "Genotype", "CellID"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            morph_cols.append(col)

    results = []
    for col in morph_cols:
        for genotype in df["Genotype"].unique():
            vals = df.loc[df["Genotype"] == genotype, col].dropna().values
            if len(vals) == 0:
                continue
            results.append(
                {
                    "Genotype": genotype,
                    "Feature": col,
                    "n_cells": int(len(vals)),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals, ddof=1)),
                }
            )
    return pd.DataFrame(results)


def correlation_matrix(df: pd.DataFrame, markers: List[str]) -> pd.DataFrame:
    marker_cols = [
        f"{m.upper()}_mean" for m in markers if f"{m.upper()}_mean" in df.columns
    ]
    if not marker_cols:
        return pd.DataFrame()
    sub_df = df[marker_cols].copy()
    corr = sub_df.corr(method="pearson")
    corr.index = [c.replace("_mean", "") for c in corr.index]
    corr.columns = [c.replace("_mean", "") for c in corr.columns]
    return corr


def plot_histograms(
    df: pd.DataFrame,
    markers: List[str],
    bins: int = 30,
    output_prefix: Optional[Path] = None,
) -> None:
    for marker in markers:
        col = f"{marker.upper()}_mean"
        if col not in df.columns:
            continue
        plt.figure()
        for genotype in df["Genotype"].unique():
            vals = df.loc[df["Genotype"] == genotype, col].dropna().values
            if len(vals) == 0:
                continue
            plt.hist(vals, bins=bins, alpha=0.5, label=genotype)
        plt.title(f"Distribution of {marker.upper()} mean intensities")
        plt.xlabel("Mean intensity")
        plt.ylabel("Number of cells")
        plt.legend()
        if output_prefix is not None:
            out_path = output_prefix.parent / f"{output_prefix.name}_hist_{marker.upper()}.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_scatter(
    df: pd.DataFrame,
    marker1: str,
    marker2: str,
    output_prefix: Optional[Path] = None,
) -> None:
    col1 = f"{marker1.upper()}_mean"
    col2 = f"{marker2.upper()}_mean"
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError("Markers not found in DataFrame")
    plt.figure()
    marker_styles = ["o", "^", "s", "x", "d"]

    for i, genotype in enumerate(df["Genotype"].unique()):
        sub = df[df["Genotype"] == genotype][[col1, col2]].dropna()
        if sub.empty:
            continue
        x = sub[col1].values
        y = sub[col2].values
        plt.scatter(
            x,
            y,
            label=genotype,
            marker=marker_styles[i % len(marker_styles)],
            alpha=0.6,
        )

    plt.title(f"{marker1.upper()} vs {marker2.upper()} mean intensities")
    plt.xlabel(f"{marker1.upper()} mean intensity")
    plt.ylabel(f"{marker2.upper()} mean intensity")
    plt.legend()

    if output_prefix is not None:
        out_path = output_prefix.parent / f"{output_prefix.name}_scatter_{marker1.upper()}_{marker2.upper()}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _infer_markers_from_df(df: pd.DataFrame) -> List[str]:
    markers = []
    for col in df.columns:
        if col.endswith("_mean"):
            base = col[:-5]
            markers.append(base)
    seen = set()
    uniq = []
    for m in markers:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


# -------------------------------------------------------------------
# Per-CSV runner (parallel worker)
# -------------------------------------------------------------------

def run_analysis_on_csv(csv_path: Path, markers_arg, no_plots, min_area, max_area):
    df = pd.read_csv(csv_path)

    # QC filter on NucleusArea
    if "NucleusArea" in df.columns:
        if min_area > 0:
            df = df[df["NucleusArea"] >= min_area]
        if max_area > 0:
            df = df[df["NucleusArea"] <= max_area]

    if df.empty:
        return

    # Markers
    if markers_arg:
        markers = [m.strip() for m in markers_arg.split(",") if m.strip()]
    else:
        markers = _infer_markers_from_df(df)

    outdir = csv_path.parent
    basename = csv_path.stem  # e.g. VUB02_per_cell_features

    # Summary stats
    stats = summary_statistics(df, markers)
    stats.to_csv(outdir / f"{basename}_summary_stats.csv", index=False)

    # Fraction positive
    frac = fraction_positive(df, markers)
    frac.to_csv(outdir / f"{basename}_fraction_positive.csv", index=False)

    # Morphology
    morph = morph_statistics(df)
    morph.to_csv(outdir / f"{basename}_morph_stats.csv", index=False)

    # Correlation
    corr = correlation_matrix(df, markers)
    corr.to_csv(outdir / f"{basename}_correlation_matrix.csv")

    # Coexpression OCT4/POU5F1 vs PAX6 (if available)
    if "OCT4_mean" in df.columns or "POU5F1_mean" in df.columns:
        m_oct = "OCT4" if "OCT4_mean" in df.columns else "POU5F1"
    else:
        m_oct = None

    if "PAX6_mean" in df.columns:
        m_pax = "PAX6"
    else:
        m_pax = None

    if m_oct is not None and m_pax is not None:
        coexp = coexpression_table(df, m_oct, m_pax)
        coexp_path = outdir / f"{basename}_coexpression_{m_oct}_{m_pax}.csv"
        coexp.to_csv(coexp_path, index=False)

    # Plots
    if not no_plots:
        prefix = outdir / basename
        plot_histograms(df, markers, output_prefix=prefix)

        if m_oct is not None and m_pax is not None:
            plot_scatter(df, m_oct, m_pax, output_prefix=prefix)


# -------------------------------------------------------------------
# CLI + parallel driver
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parallel simple_analysis over all *_per_cell_features.csv under root_dir."
    )
    parser.add_argument("--root_dir", required=True,
                        help="Root directory containing *_per_cell_features.csv files.")
    parser.add_argument("--markers", default="",
                        help="Comma-separated marker list (if omitted, inferred from *_mean columns).")
    parser.add_argument("--no_plots", action="store_true",
                        help="If set, do not generate PNG plots.")
    parser.add_argument("--min_area", type=float, default=0.0)
    parser.add_argument("--max_area", type=float, default=0.0)
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()

    root = Path(args.root_dir)
    csv_files = sorted(root.rglob("*_per_cell_features.csv"))
    if not csv_files:
        print(f"[ERROR] No *_per_cell_features.csv files found under {root}")
        return

    print(f"[INFO] Found {len(csv_files)} per-cell CSVs.")
    print(f"[INFO] Using up to {args.n_workers} worker processes.")

    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = [
            ex.submit(
                run_analysis_on_csv,
                csv,
                args.markers,
                args.no_plots,
                args.min_area,
                args.max_area,
            )
            for csv in csv_files
        ]

        for _ in tqdm(as_completed(futures),
                      total=len(futures),
                      desc="Analyzing CSV files"):
            pass


if __name__ == "__main__":
    main()
