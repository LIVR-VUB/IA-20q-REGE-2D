#!/usr/bin/env python3
import re
import csv
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Set

# ---------- Config ----------
WAVE_TOKEN = re.compile(r"^(3\d{2}|4\d{2}|5\d{2}|6\d{2}|7\d{2})$")  # 300–799
TRAILING_INDEX = re.compile(r"^(.*?)[_\- ](\d{1,3})$")  # _1, -2, etc.

KNOWN_WAVES = {
    "350","355","360","365","375","385","390","395",
    "405","420","430","440","450","460","470","480","488",
    "500","514","520","540","550","561","568",
    "580","587","590","594",
    "600","610","620","633","635","638","640","647","650","660","680",
    "700","720","750","770"
}

def tokenize(stem: str) -> List[str]:
    return [t for t in re.split(r"[\s_\-\.]+", stem) if t]

def is_label_token(tok: str) -> bool:
    return any(c.isalpha() for c in tok)

def normalize_label(tok: str) -> str:
    t = tok.upper().replace("*", "STAR")
    t = re.sub(r"[^A-Z0-9]+", "", t)
    return t if t else "UNK"

def split_sample_and_index(stem: str, preserve_index: bool = True) -> Tuple[str, Optional[int]]:
    """
    Pull off a trailing _N / -N (N=1-3 digits) as an index unless it looks like a wavelength.
    """
    if not preserve_index:
        return stem, None
    m = TRAILING_INDEX.fullmatch(stem)
    if not m:
        return stem, None
    base, idx_str = m.group(1), m.group(2)
    if idx_str in KNOWN_WAVES:
        return stem, None
    base = (base or "").strip(" _-.")
    return (base if base else stem), int(idx_str)

def find_waves(tokens: List[str]) -> List[Tuple[int, int]]:
    return [(i, int(t)) for i, t in enumerate(tokens) if WAVE_TOKEN.match(t)]

def nearest_label(tokens: List[str], wave_idx: int, max_radius: int) -> Optional[str]:
    for radius in range(1, max_radius + 1):
        for j in (wave_idx - radius, wave_idx + radius):
            if 0 <= j < len(tokens) and is_label_token(tokens[j]):
                return normalize_label(tokens[j])
    return None

def assoc_channels(tokens: List[str], max_label_distance: int) -> List[Tuple[str, int]]:
    """[(label, wavelength)] for all wavelength tokens; label = nearest token with letters (fallback L{λ})."""
    seen = set()
    channels = []
    for w_idx, w in find_waves(tokens):
        lbl = nearest_label(tokens, w_idx, max_label_distance) or f"L{w}"
        key = (lbl, w)
        if key not in seen:
            channels.append(key)
            seen.add(key)
    return channels

def sort_channels_desc(channels: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    return sorted(channels, key=lambda lw: (-lw[1], lw[0]))

def clean_sample_name(sample: str) -> str:
    cleaned = re.sub(r"[\s\-]+", "_", sample)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned

def build_sample_prefix(stem: str, tokens: List[str], labels_to_drop: Set[str]) -> str:
    """Take tokens before first wavelength, but drop tokens that normalize to channel labels."""
    waves = find_waves(tokens)
    if waves:
        first_idx = waves[0][0]
        kept = []
        for tok in tokens[:first_idx]:
            if is_label_token(tok):
                norm = normalize_label(tok)
                if norm in labels_to_drop:
                    continue
            kept.append(tok)
        prefix = " ".join(kept).strip(" _-.")
        return prefix or stem
    return stem

def build_new_name(sample: str,
                   non_dapi_sorted: List[Tuple[str, int]],
                   suffix: str,
                   dapi_wave: int,
                   trailing_idx: Optional[int],
                   preserve_index: bool) -> str:
    parts = []
    for ci, (lbl, w) in enumerate(non_dapi_sorted[:2], start=1):
        parts.append(f"c{ci}-{lbl}-{w}")
    parts.append(f"c3-DAPI-{dapi_wave}")
    base = f"{sample}__" + "__".join(parts)
    if preserve_index and trailing_idx is not None:
        base += f"__idx-{trailing_idx}"
    return base + suffix

def main():
    ap = argparse.ArgumentParser(
        description="Recursively rename .czi by wavelength (desc), force DAPI as c3, keep trailing _N as __idx-N, normalize underscores, mirror folder structure."
    )
    ap.add_argument("--indir", required=True, help="Input directory (searched recursively)")
    ap.add_argument("--outdir", required=True, help="Output directory (use same as --indir for in-place)")
    ap.add_argument("--glob", default="*.czi", help="Pattern (default: *.czi)")
    ap.add_argument("--log", default="rename_log.csv", help="CSV log path")
    ap.add_argument("--go", action="store_true", help="Apply changes (default: dry-run)")
    ap.add_argument("--assume-dapi-wave", type=int, default=405, help="Assumed DAPI wavelength")
    ap.add_argument("--max-label-distance", type=int, default=3, help="Distance to bind label to wavelength")
    ap.add_argument("--mode", choices=["copy", "move"], default="copy",
                    help="When outdir != indir: copy or move (default: copy)")
    ap.add_argument("--no-preserve-index", action="store_true",
                    help="If set, ignore trailing _N / -N numbers (do NOT append __idx-N).")
    args = ap.parse_args()

    indir = Path(args.indir).resolve()
    outdir = Path(args.outdir).resolve()
    files = sorted(indir.rglob(args.glob))
    if not files:
        print(f"No files matched under {indir}")
        return

    with open(args.log, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["old_path", "new_path", "filesize_bytes", "status"])

        renamed = 0
        for src in files:
            if not src.is_file():
                continue
            size = src.stat().st_size
            rel_parent = src.parent.relative_to(indir)

            stem_full = src.stem
            sample_base, trailing_idx = split_sample_and_index(stem_full, preserve_index=not args.no_preserve_index)
            tokens = tokenize(sample_base)

            channels = assoc_channels(tokens, args.max_label_distance)
            non_dapi_sorted = sort_channels_desc(channels)
            labels_in_channels = {lbl for (lbl, _) in channels}

            sample_raw = build_sample_prefix(sample_base, tokens, labels_in_channels)
            sample = clean_sample_name(sample_raw)

            new_filename = build_new_name(sample, non_dapi_sorted, src.suffix,
                                          args.assume_dapi_wave, trailing_idx,
                                          preserve_index=not args.no_preserve_index)

            # Destination
            if outdir == indir:
                dst = src.parent / new_filename
            else:
                dst_dir = outdir / rel_parent
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / new_filename

            if str(src) == str(dst):
                status = "SKIPPED"
                print(f"= SKIP: {src}")
            else:
                print(f"- OLD: {src.name}")
                print(f"+ NEW: {new_filename}")
                if args.go:
                    if dst.exists():
                        status = "TARGET_EXISTS_SKIP"
                        print(f"! WARN: target exists, skipping {dst.name}")
                    else:
                        if outdir == indir:
                            src.rename(dst)
                        else:
                            if args.mode == "copy":
                                shutil.copy2(src, dst)
                            else:
                                src.rename(dst)
                        renamed += 1
                        status = "RENAMED"
                else:
                    status = "DRY_RUN"
                    print("(dry-run)")

            writer.writerow([str(src), str(dst), size, status])

    print(("Done. " if args.go else "Dry-run complete. ") + f"Log: {args.log}")

if __name__ == "__main__":
    main()
