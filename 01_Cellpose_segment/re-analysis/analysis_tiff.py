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

# Optional (fast) nearest-neighbor resize
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

# Optional fallback resize
try:
    from skimage.transform import resize as sk_resize  # type: ignore
except Exception:
    sk_resize = None


# ---------------- marker → model logic ----------------

DEFAULT_MARKER_CLASSES = {
    "nuclear": ["PAX6", "OCT4", "HAND1", "GATA3", "TFAP2A", "EYA1", "DAPI"],
    "cytoplasmic": ["KRT*", "EPCAM"],
    "transmembrane": ["PMEL"],
}

DEFAULT_MARKER_MODELS = {
    "nuclear": "nuclei",
    "cytoplasmic": "cyto",
    "transmembrane": "cyto",
}

# Example file pattern:
# Nemo_20q__c1-EYA1-647__c2-PAX6-594__c3-DAPI-405__idx-1.czi
CHANNEL_MARKER_RE = re.compile(r"__c(?P<idx>\d+)-(?P<marker>[^-]+)-")


def parse_markers_from_name(name: str) -> dict[int, str]:
    markers: dict[int, str] = {}
    for m in CHANNEL_MARKER_RE.finditer(name):
        idx = int(m.group("idx")) - 1  # c1 -> 0
        marker = m.group("marker").upper()
        markers[idx] = marker
    return markers


def classify_marker(marker: str, marker_classes: dict) -> str | None:
    m_upper = marker.upper()
    for class_name, markers in marker_classes.items():
        for m in markers:
            m_u = str(m).upper()
            if m_u.endswith("*"):
                if m_upper.startswith(m_u[:-1]):
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
    cls = classify_marker(marker, marker_classes)
    if cls is not None:
        model = marker_models.get(cls)
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
        return a[np.newaxis, ...]

    if a.ndim == 3:
        ax_small = int(np.argmin(a.shape))
        if a.shape[ax_small] <= 8:
            return np.moveaxis(a, ax_small, 0)
        return a[np.newaxis, ...]

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

    if path.suffix.lower() == ".czi":
        # Fallback readers if aicsimageio fails
        try:
            import czifile  # type: ignore
            with czifile.CziFile(str(path)) as czi:
                a = czi.asarray()
            return _squeeze_and_reorder(np.asarray(a)), ch_names
        except Exception:
            pass
        try:
            from aicspylibczi import CziFile  # type: ignore
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

    arr = np.asarray(tiff.imread(str(path)))
    return _squeeze_and_reorder(arr), ch_names


def subtract_background_2d(img2d: np.ndarray, bg_value) -> np.ndarray:
    """
    Subtract constant background for Cellpose input only.
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


def _resize_labels_nearest(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Resize a *label image* using nearest-neighbor interpolation.
    """
    if mask.shape == target_shape:
        return mask

    th, tw = target_shape

    if cv2 is not None:
        out = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        return out.astype(mask.dtype, copy=False)

    if sk_resize is not None:
        out = sk_resize(
            mask,
            (th, tw),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
        return out.astype(mask.dtype, copy=False)

    raise RuntimeError("Need opencv-python (cv2) or scikit-image installed to resize masks safely.")


def mid_slice_2d(stack_or_2d: np.ndarray) -> np.ndarray:
    """
    If input is (Z,Y,X) -> take center slice.
    If input is (Y,X) -> return as-is.
    """
    if stack_or_2d.ndim == 2:
        return stack_or_2d
    if stack_or_2d.ndim == 3:
        zmid = stack_or_2d.shape[0] // 2
        return stack_or_2d[zmid]
    raise ValueError(f"Unsupported channel array ndim={stack_or_2d.ndim}")


# ---------------- Cellpose wrapper (cached + size-safe) ----------------

_MODEL_CACHE: dict[tuple[bool, str], models.CellposeModel] = {}


def _get_cellpose_model(model_identifier, gpu_ok: bool) -> models.CellposeModel:
    builtins = {"nuclei", "cyto", "cyto2", "cyto3", "bact"}

    if isinstance(model_identifier, str) and model_identifier in builtins:
        key = (gpu_ok, f"builtin::{model_identifier}")
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = models.CellposeModel(gpu=gpu_ok, model_type=model_identifier)
        return _MODEL_CACHE[key]

    if model_identifier:
        key = (gpu_ok, f"path::{str(model_identifier)}")
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = models.CellposeModel(gpu=gpu_ok, pretrained_model=model_identifier)
        return _MODEL_CACHE[key]

    key = (gpu_ok, "default::")
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = models.CellposeModel(gpu=gpu_ok)
    return _MODEL_CACHE[key]


def run_cellpose_single(
    img2d: np.ndarray,
    model_identifier,
    use_gpu: bool = True,
    flow_threshold=None,
    cellprob_threshold=None,
    diameter=None,
    force_resample: bool = True,
    verbose_resize: bool = True,
) -> np.ndarray:
    """
    Run Cellpose on single 2D image and guarantee output mask == img2d.shape.
    """
    gpu_ok = bool(use_gpu and torch.cuda.is_available())
    mdl = _get_cellpose_model(model_identifier, gpu_ok)

    kwargs = {"resample": bool(force_resample)}
    if flow_threshold is not None:
        kwargs["flow_threshold"] = float(flow_threshold)
    if cellprob_threshold is not None:
        kwargs["cellprob_threshold"] = float(cellprob_threshold)
    if diameter is not None:
        kwargs["diameter"] = float(diameter)

    masks, *_ = mdl.eval([img2d], **kwargs)
    if isinstance(masks, list):
        masks = masks[0]
    masks = np.asarray(masks)

    if masks.shape != img2d.shape:
        if verbose_resize:
            print(f"      [FIX] Mask shape {masks.shape} -> {img2d.shape} (nearest resize)")
        masks = _resize_labels_nearest(masks, img2d.shape)

    return masks


def resolve_index(idx_cfg, default_idx: int, nC: int, ch_label: str) -> int:
    """
    Resolve channel index allowing:
      - 1-based (1..nC)
      - 0-based (0..nC-1)
    """
    if nC <= 0:
        raise ValueError("No channels found in image.")

    if idx_cfg is None:
        idx = default_idx
    else:
        i = int(idx_cfg)
        if 1 <= i <= nC:
            idx = i - 1
        elif 0 <= i < nC:
            idx = i
        else:
            raise ValueError(
                f"Invalid index for {ch_label}: {idx_cfg} (nC={nC}). "
                "Use 0-based [0..nC-1] or 1-based [1..nC]."
            )

    if not (0 <= idx < nC):
        raise ValueError(f"Resolved index for {ch_label} out of range: {idx} (nC={nC}).")
    return idx


# ---------------- main ----------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Marker-aware 3-channel Cellpose segmentation with per-channel config. "
            "Z-stacks are handled by taking the CENTER slice for segmentation, "
            "but saving the raw channel image as 3D if the input is 3D. "
            "Masks are ALWAYS exported as 2D with suffix _cp.tiff."
        )
    )
    p.add_argument("config", help="Path to YAML config.")

    # CLI overrides (optional, per channel)
    for ch in ("c1", "c2", "c3"):
        p.add_argument(f"--{ch}-model", type=str, default=None)
        p.add_argument(f"--{ch}-flow", type=float, default=None)
        p.add_argument(f"--{ch}-cellprob", type=float, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    in_dir = Path(cfg["input_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    excludes = [e.lower() for e in cfg.get("exclude_name_contains", [])]

    cp_cfg = cfg.get("cellpose", {})
    use_gpu = bool(cp_cfg.get("use_gpu", True))
    global_diameter = cp_cfg.get("diameter", None)
    force_resample = bool(cp_cfg.get("force_resample", True))
    verbose_resize = bool(cp_cfg.get("verbose_resize_fix", True))

    marker_classes = cfg.get("marker_classes") or DEFAULT_MARKER_CLASSES
    marker_models = cfg.get("marker_models") or DEFAULT_MARKER_MODELS
    skip_markers = [m.upper() for m in cfg.get("skip_markers", [])]

    ch_cfg = cfg.get("channels", {})
    c1_cfg = ch_cfg.get("c1", {})
    c2_cfg = ch_cfg.get("c2", {})
    c3_cfg = ch_cfg.get("c3", {})

    # Per-channel enabled flags (default True)
    c1_enabled = bool(c1_cfg.get("enabled", True))
    c2_enabled = bool(c2_cfg.get("enabled", True))
    c3_enabled = bool(c3_cfg.get("enabled", True))

    # Channel-level fallback models (CLI > YAML)
    c1_default_model = args.c1_model if args.c1_model is not None else c1_cfg.get("model")
    c2_default_model = args.c2_model if args.c2_model is not None else c2_cfg.get("model")
    c3_default_model = args.c3_model if args.c3_model is not None else c3_cfg.get("model")

    # Thresholds (CLI > YAML)
    c1_flow   = args.c1_flow     if args.c1_flow     is not None else c1_cfg.get("flow_threshold")
    c1_cellpb = args.c1_cellprob if args.c1_cellprob is not None else c1_cfg.get("cellprob_threshold")

    c2_flow   = args.c2_flow     if args.c2_flow     is not None else c2_cfg.get("flow_threshold")
    c2_cellpb = args.c2_cellprob if args.c2_cellprob is not None else c2_cfg.get("cellprob_threshold")

    c3_flow   = args.c3_flow     if args.c3_flow     is not None else c3_cfg.get("flow_threshold")
    c3_cellpb = args.c3_cellprob if args.c3_cellprob is not None else c3_cfg.get("cellprob_threshold")

    # Optional explicit indices (YAML)
    c1_idx_cfg = c1_cfg.get("index")
    c2_idx_cfg = c2_cfg.get("index")
    c3_idx_cfg = c3_cfg.get("index")

    # Extensions
    exts = cfg.get("extensions", [".czi"])
    exts = [e.lower() for e in exts]

    # Exclude patterns for mask output files to prevent reprocessing
    output_suffixes = ("_cp.tif", "_cp.tiff")

    files: list[Path] = []
    for ext in exts:
        for p in in_dir.rglob(f"*{ext}"):
            name_lower = p.name.lower()
            # Skip if matches any user-defined excludes
            if any(x in name_lower for x in excludes):
                continue
            # Skip if file is a previously generated mask output
            if any(name_lower.endswith(suf) for suf in output_suffixes):
                continue
            files.append(p)
    files = sorted(set(files))

    if not files:
        print(f"[INFO] No files found under {in_dir} for extensions={exts}")
        return

    print(f"[INFO] Found {len(files)} files")
    print(f"[INFO] GPU available: {torch.cuda.is_available()}  |  Use GPU: {use_gpu}")
    print(f"[INFO] cellpose.force_resample={force_resample}  verbose_resize_fix={verbose_resize}")
    if global_diameter is not None:
        print(f"[INFO] cellpose.diameter={global_diameter}")
    if skip_markers:
        print(f"[INFO] skip_markers={skip_markers}")
    print("[INFO] Z handling: if channel is (Z,Y,X) -> center slice used for segmentation; mask saved as 2D.")

    for f in files:
        try:
            data, _ = load_image_any(f)  # (C,Z,Y,X) or (C,Y,X) or (Y,X)
            
            # Handle single-channel TIFF files (2D images from slicing)
            if data.ndim == 2:
                # Single 2D image - treat as single channel
                data = data[np.newaxis, ...]  # Make it (1, Y, X)
            
            nC = int(data.shape[0])

            # For single-channel files, only process c1
            if nC == 1:
                # Single channel - use c1 config only
                channels_to_process = [
                    ("c1", 0, c1_default_model, c1_flow, c1_cellpb, c1_enabled, c1_cfg),
                ]
            else:
                # Multi-channel - resolve indices as before
                c1_idx = resolve_index(c1_idx_cfg, 0, nC, "c1")
                c2_idx = resolve_index(c2_idx_cfg, 1, nC, "c2") if (c2_cfg and nC >= 2) else None
                c3_idx = resolve_index(c3_idx_cfg, 2, nC, "c3") if (c3_cfg and nC >= 3) else None

                channels_to_process = [
                    ("c1", c1_idx, c1_default_model, c1_flow, c1_cellpb, c1_enabled, c1_cfg),
                ]
                if c2_idx is not None:
                    channels_to_process.append(("c2", c2_idx, c2_default_model, c2_flow, c2_cellpb, c2_enabled, c2_cfg))
                if c3_idx is not None:
                    channels_to_process.append(("c3", c3_idx, c3_default_model, c3_flow, c3_cellpb, c3_enabled, c3_cfg))

            # Markers parsed from filename (old behavior kept)
            markers_by_idx = parse_markers_from_name(f.name)

            rel = f.relative_to(in_dir).parent
            sub_out = out_dir / rel
            sub_out.mkdir(parents=True, exist_ok=True)
            base = f.stem

            print(f"\n[FILE] {f.relative_to(in_dir)}")

            for label, idx, default_model, flow_th, cellprob_th, enabled, this_cfg in channels_to_process:
                marker = markers_by_idx.get(idx)

                if not enabled:
                    print(f"   - {label}: channel={idx}, marker={marker} -> SKIPPED (enabled=false)")
                    continue

                if marker is not None and marker.upper() in skip_markers:
                    print(f"   - {label}: channel={idx}, marker={marker} -> SKIPPED (in skip_markers)")
                    continue

                bg_val = this_cfg.get("bg_subtract", 0)
                diameter = this_cfg.get("diameter", global_diameter)

                # Raw per-channel data: (Y,X) or (Z,Y,X)
                ch_stack = data[idx]

                # --- NEW ZSTACK RULE ---
                # always segment on center slice if 3D, else on 2D
                img2d = mid_slice_2d(ch_stack)

                # model selection: marker-based -> fallback model -> nuclei
                model_for_this = infer_model_from_marker(
                    marker=marker if marker is not None else "",
                    marker_classes=marker_classes,
                    marker_models=marker_models,
                    default_model=default_model,
                ) or "nuclei"

                print(
                    f"   - {label}: channel={idx}, marker={marker}, "
                    f"bg_subtract={bg_val}, model={model_for_this}, "
                    f"flow={flow_th}, cellprob={cellprob_th}, diameter={diameter}"
                )

                img_for_cp = subtract_background_2d(img2d, bg_val)

                mask2d = run_cellpose_single(
                    img_for_cp,
                    model_for_this,
                    use_gpu=use_gpu,
                    flow_threshold=flow_th,
                    cellprob_threshold=cellprob_th,
                    diameter=diameter,
                    force_resample=force_resample,
                    verbose_resize=verbose_resize,
                )

                # extra safety
                if mask2d.shape != img2d.shape:
                    if verbose_resize:
                        print(f"      [FIX] Final safety resize {mask2d.shape} -> {img2d.shape}")
                    mask2d = _resize_labels_nearest(mask2d, img2d.shape)

                # Save mask ONLY - always 2D, always ends with _cp.tiff
                tiff.imwrite(sub_out / f"{base}_cp.tiff", mask2d.astype(np.uint16, copy=False))

            print(f"[OK] {f.relative_to(in_dir)}")

        except Exception as e:
            print(f"[WARN] Failed on {f.name}: {e}")

    print(f"\n[FINISHED] Results written under: {out_dir}\n")


if __name__ == "__main__":
    main()
