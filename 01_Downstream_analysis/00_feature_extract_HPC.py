#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mask-aware per‑cell feature extraction
=====================================

This script performs per‑cell feature extraction from multi‑channel TIFF
images where each channel is stored in its own file and may optionally
include a segmentation mask.  File names are expected to follow the
pattern used by Cellpose and similar tools, for example::

    Nemo_20q__c1-EYA1-647__c2-PAX6-594__c3-DAPI-405__idx-1_c1_img.tif
    Nemo_20q__c1-EYA1-647__c2-PAX6-594__c3-DAPI-405__idx-1_c1_cp.tif
    Nemo_20q__c1-EYA1-647__c2-PAX6-594__c3-DAPI-405__idx-1_c2_img.tif
    Nemo_20q__c1-EYA1-647__c2-PAX6-594__c3-DAPI-405__idx-1_c2_cp.tif

The part before the first ``__`` is the base name and encodes the
``CellType`` (e.g. ``Nemo_20q``).  The channel definitions (``c1-EYA1-647``
etc.) map channel indices to marker names.  For each channel there may be
an ``*_img.tif`` file containing the raw intensities and an ``*_cp.tif``
file containing a segmentation mask produced by Cellpose or similar.

Masks are used to derive nuclear and whole‑cell boundaries.  If a
mask is provided for a channel whose marker name is in ``NUCLEAR_MARKERS``,
it is taken as the nuclear segmentation.  Similarly, a mask whose
marker name is in ``CYTO_MARKERS`` is taken as the cell segmentation.
If no masks are provided, nuclei are segmented using Otsu thresholding
and watershed on an available nuclear marker channel, and cells are
derived from the nuclei using a cytoplasmic channel if available.

For each detected cell, morphological measurements and per‑channel
intensity statistics are computed.  Results are written as one CSV per
subdirectory of the input tree (preserving the relative directory
structure).  Each row in the CSV reports metadata (relative path,
image ID, cell type, genotype), nucleus and cell morphology, and
integrated/mean intensity values for every marker present.

Example usage::

    python down_analysis_parallel_HPC.py --input_dir /data/images \
        --output_dir /data/out --n_workers 4

Parallelisation:
    Each base group (one field of view with all its c1/c2/c3 files) is
    processed independently using a process pool executor.

"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
import cv2
from tqdm import tqdm

# Restrict BLAS/OpenMP threads inside workers to improve parallel performance
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Marker sets for nuclear and cytoplasmic channels.  Marker names are compared
# in uppercase.  Extend these sets as needed for your data.
NUCLEAR_MARKERS = {
    "DAPI",
    "OCT4",
    "POU5F1",
    "PAX6",
    "TFAP2A",
    "EYA1",
    "SOX2",
}

CYTO_MARKERS = {
    "EPCAM",
    "KRT",
    "CK",
    "KRT8",
    "KRT18",
    "KRT5",
    "PMEL",
}


def tifread(p: Path) -> np.ndarray:
    """Read a TIFF file into a NumPy array (first page only)."""
    with Image.open(str(p)) as im:
        try:
            im.seek(0)
        except EOFError:
            pass
        arr = np.array(im)
    return arr


def load_img(p: Path) -> np.ndarray:
    """Load an image and return a 2D float32 array."""
    img = tifread(p)
    if img.ndim != 2:
        img = np.squeeze(img)
    return img.astype(np.float32)


def _ensure_label_mask(arr: np.ndarray) -> np.ndarray:
    """Ensure that a mask is labelled (connected components labelled)."""
    arr = np.squeeze(arr).astype(np.int32)
    if arr.max() <= 1:
        arr, _ = ndi.label(arr > 0)
    return arr


def segment_nuc(nuc_img: np.ndarray) -> np.ndarray:
    """Segment nuclei from a nuclear channel using Otsu + watershed."""
    imin, imax = float(nuc_img.min()), float(nuc_img.max())
    if imax > imin:
        img_norm = (nuc_img - imin) / (imax - imin)
    else:
        img_norm = np.zeros_like(nuc_img)
    img8 = np.uint8(np.clip(img_norm * 255, 0, 255))
    try:
        _, bw = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        th = nuc_img.mean() + nuc_img.std()
        bw = np.uint8(nuc_img > th) * 255
    bw_bool = bw.astype(bool)
    labelled, num = ndi.label(bw_bool)
    sizes = np.bincount(labelled.ravel())
    remove = sizes < 50
    remove_idx = np.nonzero(remove)[0]
    for idx in remove_idx:
        labelled[labelled == idx] = 0
    bw_bool = labelled > 0
    bw_bool = ndi.binary_fill_holes(bw_bool)
    dist = cv2.distanceTransform(bw_bool.astype(np.uint8), cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    unknown = cv2.subtract(bw_bool.astype(np.uint8) * 255, sure_fg)
    markers[unknown > 0] = 0
    img_color = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(img_color, markers.astype(np.int32).copy())
    markers_ws[markers_ws == -1] = 0
    return markers_ws.astype(np.int32)


def derive_cell_labels(nuc_labels: np.ndarray, cyto_img: Optional[np.ndarray]) -> np.ndarray:
    """Derive whole‑cell labels from nuclear labels and an optional cytoplasmic image."""
    if cyto_img is not None:
        smoothed = ndi.gaussian_filter(cyto_img.astype(np.float32), sigma=2)
        imin, imax = float(smoothed.min()), float(smoothed.max())
        if imax > imin:
            img_norm = (smoothed - imin) / (imax - imin)
        else:
            img_norm = np.zeros_like(smoothed)
        img8 = np.uint8(np.clip(img_norm * 255, 0, 255))
        img_color = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        markers = nuc_labels.copy().astype(np.int32)
        markers[markers < 0] = 0
        markers_ws = cv2.watershed(img_color, markers)
        markers_ws[markers_ws == -1] = 0
        return markers_ws.astype(np.int32)
    else:
        if np.max(nuc_labels) == 0:
            return nuc_labels.astype(np.int32)
        mask_bg = nuc_labels == 0
        dist, (inds_r, inds_c) = ndi.distance_transform_edt(mask_bg, return_indices=True)
        cell_labels = nuc_labels.copy().astype(np.int32)
        cell_labels[mask_bg] = nuc_labels[inds_r[mask_bg], inds_c[mask_bg]]
        return cell_labels


def build_groups(input_root: Path) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """Group TIFF files by base name and channel definitions.

    This function parses filenames expected to follow the pattern
    ``<base>__c1-MARKER1-DYE__c2-MARKER2-DYE__...__idx-<n>_cX_<tag>.tif``.
    Files ending in ``_img`` are treated as intensity images, while files
    ending in ``_cp`` are treated as segmentation masks.  The channel id
    (e.g. ``c1``) immediately before ``_img``/``_cp`` determines which
    channel the file belongs to.  Marker names are extracted from the
    channel definitions portion of the filename.

    Parameters
    ----------
    input_root : Path
        Root directory containing TIFF files.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, Path]]]
        A nested dictionary.  Keys are base names (everything before the first
        ``__``).  For each base, the value is a dictionary with keys
        ``"imgs"`` and ``"masks"``.  ``imgs`` maps marker names (uppercase)
        to intensity image paths; ``masks`` maps marker names to mask paths.
    """
    groups: Dict[str, Dict[str, Dict[str, Path]]] = {}
    for path in input_root.rglob("*.tif"):
        stem = path.stem
        if "__" not in stem:
            continue
        base, remainder = stem.split("__", 1)
        grp = groups.setdefault(base, {"imgs": {}, "masks": {}})
        # Expect suffix to end with _img or _cp
        if not (remainder.endswith("_img") or remainder.endswith("_cp")):
            continue
        core, tag = remainder.rsplit("_", 1)
        # Extract channel id (e.g. c1) from the core
        core_parts = core.split("_")
        if not core_parts:
            continue
        chan_id = core_parts[-1]
        # Parse channel definitions to map channel id to marker name
        marker_name: Optional[str] = None
        for segment in core.split("__"):
            if segment.startswith(f"{chan_id}-"):
                marker_part = segment.split("-", 1)[1]
                marker_name = marker_part.split("-", 1)[0]
                break
        if marker_name is None:
            marker_name = chan_id
        key = marker_name.upper()
        if tag == "img":
            grp["imgs"][key] = path
        else:
            grp["masks"][key] = path
    return groups


def process_single_group(args: Tuple[str, Dict[str, Dict[str, Path]], Path]) -> List[Dict[str, float]]:
    """Process a single group of files corresponding to one field of view.

    Parameters
    ----------
    args : tuple
        A tuple ``(base, group, input_root)`` where ``base`` is the base name
        (CellType and genotype), ``group`` is a dict with keys ``imgs`` and
        ``masks`` mapping marker names to file paths, and ``input_root`` is
        the input root directory for computing relative paths.

    Returns
    -------
    List[Dict[str, float]]
        A list of dictionaries, one per detected cell, containing metadata,
        morphological measurements and per‑channel intensity statistics.
    """
    base, group, input_root = args
    # Determine a representative path for relative directory
    any_path: Optional[Path] = None
    if group["imgs"]:
        any_path = next(iter(group["imgs"].values()))
    elif group["masks"]:
        any_path = next(iter(group["masks"].values()))
    else:
        return []
    try:
        rel = any_path.parent.relative_to(input_root)
    except Exception:
        rel = Path(".")
    rel_str = str(rel).replace("\\", "/")
    image_id = base
    # CellType and Genotype from base name
    cell_type = base.split("__")[0]
    genotype = "20q_mutant" if "20q" in base.lower() else "WT"
    # Load intensity channels
    channels: Dict[str, np.ndarray] = {}
    for marker, p in group["imgs"].items():
        try:
            img = load_img(p)
        except Exception:
            continue
        channels[marker.upper()] = img
    # Load masks and assign nucleus/cell masks based on marker sets
    nuc_mask: Optional[np.ndarray] = None
    cell_mask: Optional[np.ndarray] = None
    for marker, p in group["masks"].items():
        try:
            arr = load_img(p)
        except Exception:
            continue
        arr = _ensure_label_mask(arr)
        mname = marker.upper()
        if mname in NUCLEAR_MARKERS:
            if nuc_mask is None:
                nuc_mask = arr
                continue
        if mname in CYTO_MARKERS:
            if cell_mask is None:
                cell_mask = arr
                continue
        # Fallback assignment
        if nuc_mask is None:
            nuc_mask = arr
        elif cell_mask is None:
            cell_mask = arr
    # If no nucleus mask, attempt segmentation using a nuclear marker channel
    if nuc_mask is None:
        nuc_markers = [m for m in channels if m in NUCLEAR_MARKERS]
        if nuc_markers:
            nuc_img = channels[nuc_markers[0]]
            nuc_mask = segment_nuc(nuc_img)
        else:
            # Cannot derive nuclei; skip
            return []
    # If no cell mask, derive it from nucleus and a cytoplasmic channel
    if cell_mask is None:
        cyto_markers = [m for m in channels if m in CYTO_MARKERS]
        cyto_img = channels[cyto_markers[0]] if cyto_markers else None
        cell_mask = derive_cell_labels(nuc_mask, cyto_img)
    # Ensure labels are contiguous integers
    nuc_mask = _ensure_label_mask(nuc_mask)
    cell_mask = _ensure_label_mask(cell_mask)
    # Helper to compute morphology from coordinates
    def _calc_morph_features(coords: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
        area = float(coords.shape[0])
        if area == 0:
            return (0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        centroid = coords.mean(axis=0)
        if coords.shape[0] > 1:
            centered = coords - centroid
            cov = np.cov(centered, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov, 0], [0, cov]])
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            major_axis = 4.0 * np.sqrt(max(eigvals[0], 0.0))
            minor_axis = 4.0 * np.sqrt(max(eigvals[1], 0.0)) if eigvals.size > 1 else 0.0
            ecc = np.sqrt(max(0.0, 1.0 - (minor_axis / major_axis) ** 2)) if major_axis > 0 else 0.0
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        else:
            major_axis = np.nan
            minor_axis = np.nan
            ecc = np.nan
            angle = np.nan
        return (
            area,
            float(centroid[0]),
            float(centroid[1]),
            float(major_axis),
            float(minor_axis),
            float(ecc),
            float(angle),
        )
    # Precompute cell morph features from cell_mask
    cell_props: Dict[int, Dict[str, float]] = {}
    for cid in np.unique(cell_mask):
        cid = int(cid)
        if cid == 0:
            continue
        coords = np.column_stack(np.where(cell_mask == cid))
        area, cy, cx, major, minor, ecc, angle = _calc_morph_features(coords)
        cell_props[cid] = {
            "CellArea": area,
            "CellCentroidRow": cy,
            "CellCentroidCol": cx,
            "CellMajorAxisLength": major,
            "CellMinorAxisLength": minor,
            "CellEccentricity": ecc,
            "CellOrientation": angle,
        }
    # Collect per-cell data
    rows: List[Dict[str, float]] = []
    for cell_id in np.unique(nuc_mask):
        cell_id = int(cell_id)
        if cell_id == 0:
            continue
        # nucleus coordinates
        nuc_coords = np.column_stack(np.where(nuc_mask == cell_id))
        (nuc_area, nuc_cy, nuc_cx, nuc_major, nuc_minor, nuc_ecc, nuc_angle) = _calc_morph_features(nuc_coords)
        info: Dict[str, float] = {}
        info["RelativePath"] = rel_str
        info["ImageID"] = image_id
        info["CellType"] = cell_type
        info["Genotype"] = genotype
        info["CellID"] = cell_id
        # Nucleus metrics
        info["NucleusArea"] = nuc_area
        info["CentroidRow"] = nuc_cy
        info["CentroidCol"] = nuc_cx
        info["NucleusMajorAxisLength"] = nuc_major
        info["NucleusMinorAxisLength"] = nuc_minor
        info["NucleusEccentricity"] = nuc_ecc
        info["NucleusOrientation"] = nuc_angle
        # Cell metrics
        cell_prop = cell_props.get(cell_id)
        if cell_prop:
            cell_area = cell_prop.get("CellArea", np.nan)
            info["CellArea"] = cell_area
            info["CellMajorAxisLength"] = cell_prop.get("CellMajorAxisLength", np.nan)
            info["CellMinorAxisLength"] = cell_prop.get("CellMinorAxisLength", np.nan)
            info["CellEccentricity"] = cell_prop.get("CellEccentricity", np.nan)
            info["CellOrientation"] = cell_prop.get("CellOrientation", np.nan)
            info["NucleusToCellAreaRatio"] = (
                nuc_area / cell_area if cell_area and cell_area > 0 else np.nan
            )
        else:
            info["CellArea"] = np.nan
            info["CellMajorAxisLength"] = np.nan
            info["CellMinorAxisLength"] = np.nan
            info["CellEccentricity"] = np.nan
            info["CellOrientation"] = np.nan
            info["NucleusToCellAreaRatio"] = np.nan
        # Compute per-marker intensity stats
        nuc_region_mask = nuc_mask == cell_id
        cell_region_mask = cell_mask == cell_id
        for m, img in channels.items():
            region = nuc_region_mask if m in NUCLEAR_MARKERS else cell_region_mask
            if region.any():
                px = img[region]
                info[f"{m}_int"] = float(px.sum())
                info[f"{m}_mean"] = float(px.mean())
            else:
                info[f"{m}_int"] = 0.0
                info[f"{m}_mean"] = 0.0
        rows.append(info)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Per‑cell feature extraction for multi‑channel TIFFs with optional masks.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=4)
    args = parser.parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    groups = build_groups(input_root)
    if not groups:
        print(f"No TIFF files found under {input_root}")
        return
    args_list = [(base, group, input_root) for base, group in groups.items()]
    all_rows: List[Dict[str, float]] = []
    if args.n_workers <= 1:
        for arg in tqdm(args_list, desc="Processing", unit="image"):
            rows = process_single_group(arg)
            all_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {pool.submit(process_single_group, arg): arg[0] for arg in args_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="image"):
                try:
                    rows = future.result()
                except Exception as exc:
                    base = futures[future]
                    print(f"Error processing {base}: {exc}")
                    rows = []
                all_rows.extend(rows)
    if not all_rows:
        print("No cells were detected in any image")
        return
    df = pd.DataFrame(all_rows)
    # Write per-directory CSVs grouped by relative path
    for rel_dir, sub_df in df.groupby("RelativePath"):
        out_folder = output_root / Path(rel_dir)
        out_folder.mkdir(parents=True, exist_ok=True)
        name = out_folder.name or "root"
        csv_path = out_folder / f"{name}_per_cell_features.csv"
        sub_df.to_csv(csv_path, index=False)
        print(f"[OK] {len(sub_df)} rows → {csv_path}")


if __name__ == "__main__":
    main()
