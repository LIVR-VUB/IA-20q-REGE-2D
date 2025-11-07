#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
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


# ---------------- utils ----------------

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


def run_cellpose_single(
    img2d: np.ndarray,
    model_identifier: str | None,
    use_gpu: bool = True,
    flow_threshold: float | None = None,
    cellprob_threshold: float | None = None,
) -> np.ndarray:
    gpu_ok = bool(use_gpu and torch.cuda.is_available())
    if model_identifier:
        mdl = models.CellposeModel(model_type=model_identifier, gpu=gpu_ok)
    else:
        mdl = models.CellposeModel(gpu=gpu_ok)

    kwargs = {}
    if flow_threshold is not None:
        kwargs["flow_threshold"] = float(flow_threshold)
    if cellprob_threshold is not None:
        kwargs["cellprob_threshold"] = float(cellprob_threshold)

    masks, *_ = mdl.eval(img2d, **kwargs)
    return masks


def pick_channel_indices(
    nC: int,
    ch_names: list[str] | None,
    filename: str,
    cfg_epcam_idx: int | None,
    cfg_oct4_idx: int | None,
):
    # explicit indices from YAML (0-based; accept 1-based quietly)
    if cfg_epcam_idx is not None and cfg_oct4_idx is not None:
        ei, oi = cfg_epcam_idx, cfg_oct4_idx
        if ei >= 1 and oi >= 1 and (ei - 1) in range(nC) and (oi - 1) in range(nC):
            return ei - 1, oi - 1
        return ei, oi

    # channel names
    if ch_names:
        low = [s.lower() for s in ch_names]
        try:
            ei = next(i for i, s in enumerate(low) if "epcam" in s)
            oi = next(i for i, s in enumerate(low) if "oct4" in s)
            return ei, oi
        except StopIteration:
            pass

    # filename hint (order of tokens)
    s = filename.lower()
    pos_e = s.find("epcam")
    pos_o = s.find("oct4")
    if nC >= 2 and pos_e != -1 and pos_o != -1:
        return (0, 1) if pos_e < pos_o else (1, 0)

    # defaults
    if nC == 2:
        return 0, 1
    if nC >= 3:
        return 1, 2
    raise ValueError("Could not determine channel indices for EPCAM and OCT4.")


# ---------------- main ----------------

def parse_args():
    p = argparse.ArgumentParser(description="EPCAM–OCT4 segmentation (per-channel models & thresholds).")
    p.add_argument("config", help="Path to YAML config.")
    # CLI overrides (optional)
    p.add_argument("--epcam-model", type=str, default=None)
    p.add_argument("--oct4-model",  type=str, default=None)
    p.add_argument("--epcam-flow", type=float, default=None)
    p.add_argument("--epcam-cellprob", type=float, default=None)
    p.add_argument("--oct4-flow", type=float, default=None)
    p.add_argument("--oct4-cellprob", type=float, default=None)
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

    # per-channel config from YAML (can be overridden by CLI)
    ch_cfg = cfg.get("channels", {})
    epcam_cfg = ch_cfg.get("epcam", {})
    oct4_cfg  = ch_cfg.get("oct4",  {})

    # models (CLI > YAML)
    epcam_model = args.epcam_model if args.epcam_model is not None else epcam_cfg.get("model")
    oct4_model  = args.oct4_model  if args.oct4_model  is not None else oct4_cfg.get("model", "nuclei")

    # thresholds (CLI > YAML; may be None)
    ep_flow   = args.epcam_flow     if args.epcam_flow     is not None else epcam_cfg.get("flow_threshold")
    ep_cellpb = args.epcam_cellprob if args.epcam_cellprob is not None else epcam_cfg.get("cellprob_threshold")
    oc_flow   = args.oct4_flow      if args.oct4_flow      is not None else oct4_cfg.get("flow_threshold")
    oc_cellpb = args.oct4_cellprob  if args.oct4_cellprob  is not None else oct4_cfg.get("cellprob_threshold")

    # save switches (CLI > YAML)
    save_cfg = cfg.get("save", {})
    save_proj = args.proj if args.proj is not None else bool(save_cfg.get("projections", False))
    save_zstk = args.zstk if args.zstk is not None else bool(save_cfg.get("zstack", False))

    # optional explicit indices
    ep_idx = epcam_cfg.get("index")
    oc_idx = oct4_cfg.get("index")

    files = [f for f in in_dir.rglob("*.czi") if not any(x in f.name.lower() for x in excludes)]
    if not files:
        print(f"[INFO] No .czi files found under {in_dir}")
        return

    print(f"[INFO] Found {len(files)} files")
    print(f"[INFO] GPU available: {torch.cuda.is_available()}  |  Use GPU: {use_gpu}")
    print(f"[INFO] save.projections={save_proj}  save.zstack={save_zstk}")
    print(f"[INFO] EPCAM model={epcam_model}  thresholds(flow,cellprob)=({ep_flow},{ep_cellpb})")
    print(f"[INFO] OCT4  model={oct4_model}  thresholds(flow,cellprob)=({oc_flow},{oc_cellpb})")

    for f in files:
        try:
            data, ch_names = load_image_any(f)  # (C,Z,Y,X) or (C,Y,X)
            nC = int(data.shape[0])

            ei, oi = pick_channel_indices(nC, ch_names, f.name, ep_idx, oc_idx)

            epcam = data[ei]
            oct4  = data[oi]

            epcam_img = normalize_image(z_project(epcam, z_strategy))
            oct4_img  = normalize_image(z_project(oct4,  z_strategy))

            # segment with per-channel models & thresholds
            epcam_masks = run_cellpose_single(epcam_img, epcam_model, use_gpu, ep_flow, ep_cellpb)
            oct4_masks  = run_cellpose_single(oct4_img,  oct4_model,  use_gpu, oc_flow, oc_cellpb)

            # write to mirrored structure
            rel = f.relative_to(in_dir).parent
            sub_out = out_dir / rel
            sub_out.mkdir(parents=True, exist_ok=True)

            base = f.stem
            tiff.imwrite(sub_out / f"{base}_epcam_cp.tif", epcam_masks.astype(np.uint16))
            tiff.imwrite(sub_out / f"{base}_oct4_cp.tif",  oct4_masks.astype(np.uint16))

            # optional: projected images and/or full z-stacks
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
