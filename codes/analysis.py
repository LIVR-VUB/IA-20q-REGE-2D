#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import yaml
import numpy as np
import tifffile as tiff
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="bfio")
warnings.filterwarnings("ignore", category=ImportWarning, module="aicsimageio")

try:
    from aicsimageio import AICSImage
except Exception:
    AICSImage = None

from cellpose import models


def _squeeze_and_reorder(a: np.ndarray) -> np.ndarray:
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
                f"Failed to read CZI: {path.name}. Install one of: aicsimageio, czifile, aicspylibczi."
            )
    arr = np.asarray(tiff.imread(str(path)))
    return _squeeze_and_reorder(arr), ch_names


def z_project(stack: np.ndarray, strategy: str = "max") -> np.ndarray:
    if stack.ndim == 2:
        return stack
    return np.median(stack, axis=0) if strategy == "median" else np.max(stack, axis=0)


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    p1, p99 = np.percentile(img, (1, 99))
    if np.isfinite(p1) and np.isfinite(p99) and p99 > p1:
        img = (img - p1) / (p99 - p1)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(img, 0.0, 1.0)


def _to_u16(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def run_cellpose_single(img2d: np.ndarray, model_identifier: str | None, use_gpu: bool = True) -> np.ndarray:
    gpu_ok = bool(use_gpu and torch.cuda.is_available())
    if model_identifier:
        mdl = models.CellposeModel(model_type=model_identifier, gpu=gpu_ok)
    else:
        mdl = models.CellposeModel(gpu=gpu_ok)
    masks, *_ = mdl.eval(img2d)
    return masks


def pick_channel_indices(
    nC: int,
    ch_names: list[str] | None,
    filename: str,
    cfg_channels: dict | None,
):
    # 1) explicit indices from YAML (0-based; if 1-based is given, auto-fix)
    if cfg_channels:
        ei = cfg_channels.get("epcam_index")
        oi = cfg_channels.get("oct4_index")
        if ei is not None and oi is not None:
            # accept 1-based silently
            if ei >= 1 and oi >= 1 and ei <= nC and oi <= nC and (ei - 1) in range(nC) and (oi - 1) in range(nC):
                return ei - 1, oi - 1
            return ei, oi

    # 2) channel names from CZI metadata
    if ch_names:
        low = [s.lower() for s in ch_names]
        try:
            ei = next(i for i, s in enumerate(low) if "epcam" in s)
            oi = next(i for i, s in enumerate(low) if "oct4" in s)
            return ei, oi
        except StopIteration:
            pass

    # 3) infer order from filename (e.g., "... EPCAM 488 OCT4 647 ...")
    s = filename.lower()
    pos_e = s.find("epcam")
    pos_o = s.find("oct4")
    if nC >= 2 and pos_e != -1 and pos_o != -1:
        return (0, 1) if pos_e < pos_o else (1, 0)

    # 4) robust defaults
    if nC == 2:
        return 0, 1
    if nC >= 3:
        return 1, 2
    raise ValueError("Could not determine channel indices for EPCAM and OCT4.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_epcam_oct4_pipeline.py /path/to/config.yaml")
        sys.exit(1)

    cfg_path = Path(sys.argv[1])
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    in_dir = Path(cfg["input_dir"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    z_strategy = cfg.get("z_strategy", "max")
    excludes = [e.lower() for e in cfg.get("exclude_name_contains", [])]

    cp_cfg = cfg.get("cellpose", {})
    use_gpu = bool(cp_cfg.get("use_gpu", True))
    epcam_model = cp_cfg.get("epcam_model")

    save_cfg = cfg.get("save", {})
    save_proj = bool(save_cfg.get("projections", False))
    save_zstk = bool(save_cfg.get("zstack", False))

    ch_cfg = cfg.get("channels", {})

    files = [f for f in in_dir.rglob("*.czi") if not any(x in f.name.lower() for x in excludes)]
    if not files:
        print(f"[INFO] No .czi files found under {in_dir}")
        return

    print(f"[INFO] Found {len(files)} files")
    print(f"[INFO] GPU available: {torch.cuda.is_available()}  |  Use GPU: {use_gpu}")

    for f in files:
        try:
            data, ch_names = load_image_any(f)   # (C,Z,Y,X) or (C,Y,X)
            nC = int(data.shape[0])

            ei, oi = pick_channel_indices(nC, ch_names, f.name, ch_cfg)

            epcam = data[ei]
            oct4  = data[oi]

            epcam_img = normalize_image(z_project(epcam, z_strategy))
            oct4_img  = normalize_image(z_project(oct4,  z_strategy))

            epcam_masks = run_cellpose_single(epcam_img, epcam_model, use_gpu)
            oct4_masks  = run_cellpose_single(oct4_img,  "nuclei",    use_gpu)

            rel = f.relative_to(in_dir).parent
            sub_out = out_dir / rel
            sub_out.mkdir(parents=True, exist_ok=True)

            base = f.stem
            tiff.imwrite(sub_out / f"{base}_epcam_cp.tif", epcam_masks.astype(np.uint16))
            tiff.imwrite(sub_out / f"{base}_oct4_cp.tif",  oct4_masks.astype(np.uint16))

            # Optional: save raw projections and/or full z-stacks to TIFF (for quant)
            if save_proj:
                tiff.imwrite(sub_out / f"{base}_epcam_proj.tif", _to_u16(epcam_img))
                tiff.imwrite(sub_out / f"{base}_oct4_proj.tif",  _to_u16(oct4_img))
            if save_zstk and epcam.ndim >= 3:
                tiff.imwrite(sub_out / f"{base}_epcam_z.tif", _to_u16(normalize_image(epcam)))
                tiff.imwrite(sub_out / f"{base}_oct4_z.tif",  _to_u16(normalize_image(oct4)))

            print(f"[OK] {f.relative_to(in_dir)}")

        except Exception as e:
            print(f"[WARN] Failed on {f.name}: {e}")

    print(f"\n[FINISHED] Results written under: {out_dir}\n")


if __name__ == "__main__":
    main()
