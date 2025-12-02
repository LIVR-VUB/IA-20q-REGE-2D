#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import warnings
import re

import numpy as np
import tifffile as tiff
import torch
import yaml

warnings.filterwarnings("ignore", category=UserWarning, module="bfio")
warnings.filterwarnings("ignore", category=ImportWarning, module="aicsimageio")

try:
    from aicsimageio import AICSImage
except Exception:
    AICSImage = None

from cellpose import models


# ---------------- marker → model logic ----------------

# Default biological mapping (can be overridden in YAML)
DEFAULT_MARKER_CLASSES = {
    "nuclear": [
        "PAX6",
        "OCT4",
        "HAND1",
        "GATA3",
        "TFAP2A",
        "EYA1",
        "DAPI",
    ],
    "cytoplasmic": [
        "KRT*",   # wildcard: any KRTxx
        "EPCAM",
    ],
    "transmembrane": [
        "PMEL",
    ],
}

DEFAULT_MARKER_MODELS = {
    "nuclear": "nuclei",
    "cytoplasmic": "cyto",
    "transmembrane": "cyto",
}

# Example file pattern:
# Nemo_20q__c1-EYA1-647__c2-PAX6-594__c3-DAPI-405__idx-1.czi
# We extract: c1->EYA1, c2->PAX6, c3->DAPI
CHANNEL_MARKER_RE = re.compile(r"__c(?P<idx>\d+)-(?P<marker>[^-]+)-")


def parse_markers_from_name(name: str) -> dict[int, str]:
    """
    Parse channel markers from a filename.
    Returns a dict mapping 0-based channel index -> marker name (uppercased).
    """
    markers = {}
    for m in CHANNEL_MARKER_RE.finditer(name):
        idx = int(m.group("idx")) - 1  # c1 -> index 0
        marker = m.group("marker").upper()
        markers[idx] = marker
    return markers


def classify_marker(marker: str, marker_classes: dict) -> str | None:
    """
    Given a marker name and marker_classes dict, return class name
    ('nuclear', 'cytoplasmic', etc.) or None if not matched.

    Supports wildcard entries in marker_classes: 'KRT*' will match KRT14, KRT17, etc.
    """
    m_upper = marker.upper()
    for class_name, markers in marker_classes.items():
        for m in markers:
            m_u = str(m).upper()
            if m_u.endswith("*"):
                # wildcard prefix match, e.g. 'KRT*'
                prefix = m_u[:-1]
                if m_upper.startswith(prefix):
                    return class_name
            else:
                if m_upper == m_u:
                    return class_name
    return None


def infer_model_from_marker(
    marker: str,
    marker_classes: dict,
    marker_models: dict,
    default_model: str | None = None,
) -> str | None:
    """
    Given a marker name, infer which Cellpose model to use based on
    marker_classes and marker_models. Fallback to default_model if needed.
    """
    class_name = classify_marker(marker, marker_classes)
    if class_name is not None:
        model = marker_models.get(class_name)
        if model is not None:
            return model
    return default_model


# ---------------- utils ----------------

def _squeeze_and_reorder(a: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary CZI/TIFF shapes into something like (C, Z, Y, X) or (C, Y, X).
    """
    while a.ndim > 2 and 1 in a.shape:
        a = np.squeeze(a, axis=np.where(np.array(a.shape) == 1)[0][0])

    if a.ndim < 3:
        # assume (Y, X) -> add channel axis
        return a[np.newaxis, ...]

    if a.ndim == 3:
        # one axis is small: assume that is channel
        ax_small = int(np.argmin(a.shape))
        if a.shape[ax_small] <= 8:
            return np.moveaxis(a, ax_small, 0)
        # else: treat as (Z, Y, X) single-channel
        return a[np.newaxis, ...]

    # ndim >= 4
    for ax in np.argsort(a.shape):
        if a.shape[ax] <= 8:
            a = np.moveaxis(a, ax, 0)
            break

    while a.ndim > 4:
        a = np.max(a, axis=1)

    return a


def load_image_any(path: Path):
    """
    Load CZI or TIFF. Returns (array, channel_names).
    Output array is roughly (C, Z, Y, X) or (C, Y, X).
    """
    ch_names = None

    # Try aicsimageio first
    if AICSImage is not None:
        try:
            img = AICSImage(str(path))
            data = img.get_image_data("CZYX")
            try:
                names = img.get_channel_names()
                if names:
                    ch_names = [str(c) if c is not None else "" for c in names]
            except Exception:
                pass
            return data, ch_names
        except Exception:
            pass

    # Fallback: direct CZI reading
    if path.suffix.lower() == ".czi":
        try:
            import czifile
            with czifile.CziFile(str(path)) as czi:
                a = czi.asarray()
            return _squeeze_and_reorder(np.asarray(a)), ch_names
        except Exception:
            pass
        try:
            from aicspylibczi import CziFile
            with CziFile(str(path)) as czi:
                a = czi.read_image()
                if isinstance(a, dict) and "image" in a:
                    a = a["image"]
            return _squeeze_and_reorder(np.asarray(a)), ch_names
        except Exception:
            raise RuntimeError(
                f"Failed to read CZI: {path.name}. Install one of: "
                "aicsimageio, czifile, aicspylibczi."
            )

    # Fallback: TIFF
    arr = np.asarray(tiff.imread(str(path)))
    return _squeeze_and_reorder(arr), ch_names


def z_project(stack: np.ndarray, strategy: str = "max") -> np.ndarray:
    """
    If stack is (Y, X), returns it unchanged.
    If stack is (Z, Y, X), performs max/median over Z.
    """
    if stack.ndim == 2:
        return stack
    if strategy == "median":
        return np.median(stack, axis=0)
    return np.max(stack, axis=0)


def subtract_background_2d(img2d: np.ndarray, bg_value) -> np.ndarray:
    """
    Subtract a constant background (like Fiji's 'Subtract...') from a 2D image.
    Only used for Cellpose input; saved images stay raw.

    img2d: (Y, X), any numeric dtype
    bg_value: float/int or None/0 for no subtraction
    """
    if bg_value is None:
        return img2d.astype(np.float32, copy=False)

    bg = float(bg_value)
    if bg <= 0:
        return img2d.astype(np.float32, copy=False)

    arr = img2d.astype(np.float32, copy=False)
    arr -= bg
    arr[arr < 0] = 0.0
    return arr


def run_cellpose_single(
    img2d: np.ndarray,
    model_identifier,
    use_gpu: bool = True,
    flow_threshold=None,
    cellprob_threshold=None,
) -> np.ndarray:
    """
    Run Cellpose on a single 2D image.
    Compatible with Cellpose v4: built-ins via model_type,
    custom paths via pretrained_model.
    """
    gpu_ok = bool(use_gpu and torch.cuda.is_available())

    builtins = {"nuclei", "cyto", "cyto2", "cyto3", "bact"}

    if isinstance(model_identifier, str) and model_identifier in builtins:
        mdl = models.CellposeModel(gpu=gpu_ok, model_type=model_identifier)
    elif model_identifier:
        # treat as path to custom model
        mdl = models.CellposeModel(gpu=gpu_ok, pretrained_model=model_identifier)
    else:
        mdl = models.CellposeModel(gpu=gpu_ok)

    kwargs = {}
    if flow_threshold is not None:
        kwargs["flow_threshold"] = float(flow_threshold)
    if cellprob_threshold is not None:
        kwargs["cellprob_threshold"] = float(cellprob_threshold)

    masks, *_ = mdl.eval(img2d, **kwargs)
    return masks


def resolve_index(idx_cfg, default_idx: int, nC: int, ch_label: str) -> int:
    """
    Resolve channel index allowing:
      - 1-based indexing (1..nC)
      - 0-based indexing (0..nC-1)
    """
    if nC <= 0:
        raise ValueError("No channels found in image.")

    if idx_cfg is None:
        idx = default_idx
    else:
        i = int(idx_cfg)
        if 1 <= i <= nC:
            idx = i - 1      # treat as 1-based
        elif 0 <= i < nC:
            idx = i          # treat as 0-based
        else:
            raise ValueError(
                f"Invalid index for {ch_label}: {idx_cfg} (nC={nC}). "
                "Use 0-based [0..nC-1] or 1-based [1..nC]."
            )

    if not (0 <= idx < nC):
        raise ValueError(
            f"Resolved index for {ch_label} out of range: {idx} (nC={nC})."
        )
    return idx


# ---------------- main ----------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generic 3-channel Cellpose segmentation (c1, c2, c3) "
            "with automatic model selection based on marker names "
            "and optional background subtraction for Cellpose only."
        )
    )
    p.add_argument("config", help="Path to YAML config.")

    # CLI overrides (optional, per channel)
    for ch in ("c1", "c2", "c3"):
        p.add_argument(f"--{ch}-model", type=str, default=None)
        p.add_argument(f"--{ch}-flow", type=float, default=None)
        p.add_argument(f"--{ch}-cellprob", type=float, default=None)

    p.add_argument("--projections", dest="proj", action="store_true")
    p.add_argument("--no-projections", dest="proj", action="store_false")
    p.add_argument("--zstack", dest="zstk", action="store_true")
    p.add_argument("--no-zstack", dest="zstk", action="store_false")
    p.set_defaults(proj=None, zstk=None)

    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    in_dir = Path(cfg["input_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    z_strategy = cfg.get("z_strategy", "max")
    excludes = [e.lower() for e in cfg.get("exclude_name_contains", [])]

    cp_cfg = cfg.get("cellpose", {})
    use_gpu = bool(cp_cfg.get("use_gpu", True))

    # Marker → class and class → model mappings
    marker_classes = cfg.get("marker_classes") or DEFAULT_MARKER_CLASSES
    marker_models = cfg.get("marker_models") or DEFAULT_MARKER_MODELS

    # Markers to skip entirely (no segmentation)
    skip_markers = [m.upper() for m in cfg.get("skip_markers", [])]

    # Per-channel config from YAML
    ch_cfg = cfg.get("channels", {})
    c1_cfg = ch_cfg.get("c1", {})
    c2_cfg = ch_cfg.get("c2", {})
    c3_cfg = ch_cfg.get("c3", {})

    # Per-channel enabled flags (default True)
    c1_enabled = bool(c1_cfg.get("enabled", True))
    c2_enabled = bool(c2_cfg.get("enabled", True))
    c3_enabled = bool(c3_cfg.get("enabled", True))

    # Channel-level default models (CLI > YAML; may be None)
    c1_default_model = args.c1_model if args.c1_model is not None else c1_cfg.get("model")
    c2_default_model = args.c2_model if args.c2_model is not None else c2_cfg.get("model")
    c3_default_model = args.c3_model if args.c3_model is not None else c3_cfg.get("model")

    # Thresholds (CLI > YAML; may be None)
    c1_flow   = args.c1_flow     if args.c1_flow     is not None else c1_cfg.get("flow_threshold")
    c1_cellpb = args.c1_cellprob if args.c1_cellprob is not None else c1_cfg.get("cellprob_threshold")

    c2_flow   = args.c2_flow     if args.c2_flow     is not None else c2_cfg.get("flow_threshold")
    c2_cellpb = args.c2_cellprob if args.c2_cellprob is not None else c2_cfg.get("cellprob_threshold")

    c3_flow   = args.c3_flow     if args.c3_flow     is not None else c3_cfg.get("flow_threshold")
    c3_cellpb = args.c3_cellprob if args.c3_cellprob is not None else c3_cfg.get("cellprob_threshold")

    # Save switches (CLI > YAML)
    save_cfg = cfg.get("save", {})
    save_proj = args.proj if args.proj is not None else bool(save_cfg.get("projections", False))
    save_zstk = args.zstk if args.zstk is not None else bool(save_cfg.get("zstack", False))

    # Optional explicit indices
    c1_idx_cfg = c1_cfg.get("index")
    c2_idx_cfg = c2_cfg.get("index")
    c3_idx_cfg = c3_cfg.get("index")

    files = [
        f for f in in_dir.rglob("*.czi")
        if not any(x in f.name.lower() for x in excludes)
    ]
    if not files:
        print(f"[INFO] No .czi files found under {in_dir}")
        return

    print(f"[INFO] Found {len(files)} files")
    print(f"[INFO] GPU available: {torch.cuda.is_available()}  |  Use GPU: {use_gpu}")
    print(f"[INFO] save.projections={save_proj}  save.zstack={save_zstk}")
    print(f"[INFO] Default models (if no marker-based mapping is found):")
    print(f"       c1: {c1_default_model}  c2: {c2_default_model}  c3: {c3_default_model}")
    if skip_markers:
        print(f"[INFO] skip_markers={skip_markers}")

    for f in files:
        try:
            data, ch_names = load_image_any(f)  # (C,Z,Y,X) or (C,Y,X)
            nC = int(data.shape[0])

            # Resolve indices; defaults: c1->0, c2->1, c3->2
            c1_idx = resolve_index(c1_idx_cfg, 0, nC, "c1")

            if c2_cfg and nC >= 2:
                c2_idx = resolve_index(c2_idx_cfg, 1, nC, "c2")
            else:
                c2_idx = None

            if c3_cfg and nC >= 3:
                c3_idx = resolve_index(c3_idx_cfg, 2, nC, "c3")
            else:
                c3_idx = None

            channels_to_process = []
            channels_to_process.append(
                ("c1", c1_idx, c1_default_model, c1_flow, c1_cellpb, c1_enabled)
            )

            if c2_idx is not None:
                channels_to_process.append(
                    ("c2", c2_idx, c2_default_model, c2_flow, c2_cellpb, c2_enabled)
                )
            if c3_idx is not None:
                channels_to_process.append(
                    ("c3", c3_idx, c3_default_model, c3_flow, c3_cellpb, c3_enabled)
                )

            # Parse markers from filename
            markers_by_idx = parse_markers_from_name(f.name)

            results = {}
            print(f"\n[FILE] {f.relative_to(in_dir)}")

            for (
                label,
                idx,
                default_model,
                flow_th,
                cellprob_th,
                enabled,
            ) in channels_to_process:
                marker = markers_by_idx.get(idx)

                # 1) Skip if channel disabled in YAML
                if not enabled:
                    print(f"   - {label}: channel={idx}, marker={marker} -> SKIPPED (enabled=false)")
                    continue

                # 2) Skip if marker is in skip_markers
                if marker is not None and marker.upper() in skip_markers:
                    print(f"   - {label}: channel={idx}, marker={marker} -> SKIPPED (in skip_markers)")
                    continue

                # 3) Get the correct channel config (for bg_subtract, etc.)
                if label == "c1":
                    this_cfg = c1_cfg
                elif label == "c2":
                    this_cfg = c2_cfg
                else:
                    this_cfg = c3_cfg

                bg_val = this_cfg.get("bg_subtract", 0)

                # 4) Raw stack from image
                ch_stack = data[idx]   # (Y,X) or (Z,Y,X)

                # 5) Raw projection (this is what we'll save to *_img.tif)
                proj_raw = z_project(ch_stack, z_strategy)

                # 6) Background-subtracted projection for Cellpose only
                proj_for_cp = subtract_background_2d(proj_raw, bg_val)

                # 7) Decide which model to use: marker-based → default
                model_for_this = infer_model_from_marker(
                    marker=marker if marker is not None else "",
                    marker_classes=marker_classes,
                    marker_models=marker_models,
                    default_model=default_model,
                )

                # If still None, fall back to "nuclei" as a generic default
                if model_for_this is None:
                    model_for_this = "nuclei"

                print(
                    f"   - {label}: channel={idx}, marker={marker}, "
                    f"bg_subtract={bg_val}, model={model_for_this}"
                )

                masks = run_cellpose_single(
                    proj_for_cp,
                    model_for_this,
                    use_gpu,
                    flow_th,
                    cellprob_th,
                )

                results[label] = {
                    "masks": masks,
                    "proj_raw": proj_raw,     # raw projection (saved)
                    "full_stack": ch_stack,   # raw stack
                    "marker": marker,
                    "model": model_for_this,
                }

            # Write outputs to mirrored structure
            rel = f.relative_to(in_dir).parent
            sub_out = out_dir / rel
            sub_out.mkdir(parents=True, exist_ok=True)

            base = f.stem  # keeps the original filename (e.g. Nemo_20q...)

            for label, res in results.items():
                masks = res["masks"]
                proj_raw = res["proj_raw"]
                full_stack = res["full_stack"]

                # 1) Save the (raw) projected channel image (NO background subtraction)
                tiff.imwrite(
                    sub_out / f"{base}_{label}_img.tif",
                    proj_raw,
                )

                # 2) Save the segmentation mask
                tiff.imwrite(
                    sub_out / f"{base}_{label}_cp.tif",
                    masks.astype(np.uint16),
                )

                # 3) Optional: projections (same as *_img for 2D)
                if save_proj:
                    tiff.imwrite(
                        sub_out / f"{base}_{label}_proj.tif",
                        proj_raw,
                    )

                # 4) Optional: full Z-stack export, if present (raw intensities)
                if save_zstk and full_stack.ndim >= 3:
                    tiff.imwrite(
                        sub_out / f"{base}_{label}_z.tif",
                        full_stack,
                    )

            print(f"[OK] {f.relative_to(in_dir)}")

        except Exception as e:
            print(f"[WARN] Failed on {f.name}: {e}")

    print(f"\n[FINISHED] Results written under: {out_dir}\n")


if __name__ == "__main__":
    main()
