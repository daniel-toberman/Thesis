import os
import csv
import shutil
import re
from pathlib import Path
from typing import List, Tuple

# ====== CONFIGURE THESE ======
ROOT = r"D:\RealMAN"                    # contains extracted/, train/, val/, test/
DATA_DIR = "extracted"                  # where the audio actually lives
SPLIT_DIRS = ["train", "val", "test"]   # where CSVs live
CSV_NAME_PATTERN = "08"                 # csv filenames must include this substring
DEST_NAME = "RealMAN_dataset_T60_08"    # output folder name alongside ROOT
COPY_CSVS_TOO = True                    # copy the used CSVs into the destination
CSV_FILENAME_COLUMN = "filename"        # column that holds the base file path
PROGRESS_EVERY = 500                    # print a progress line every N copied/skipped files

# For EVERY CSV row, try each of these folders (by replacing the second path segment)
MIRROR_FOLDERS = ["dp_speech", "ma_noise", "ma_speech", "ma_noisy_speech"]

# Diagnostics: for the first N rows, show source folder existence and counts
DEBUG_FIRST_N_ROWS = 3
ROW_PROGRESS_EVERY = 200
# =======================================

root = Path(ROOT)
data_root = root / DATA_DIR                     # e.g., D:\RealMAN\extracted
dest_root = root / DEST_NAME
dest_extracted = dest_root / DATA_DIR           # e.g., D:\RealMAN\RealMAN_dataset_T60_08\extracted
dest_extracted.mkdir(parents=True, exist_ok=True)

copied = []             # (csv, src, dst)
skipped_exists = []     # (csv, src, dst)
missing_channels = []   # (csv, base_dir_rel, base_stem, channel, tried_paths)
no_candidates = []      # (csv, csv_filename_value)

def read_csv_rows(csv_path: Path) -> List[str]:
    rows = []
    for enc in ("utf-8", None):
        try:
            with csv_path.open("r", newline="", encoding=enc) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    v = (row.get(CSV_FILENAME_COLUMN) or "").strip()
                    if v:
                        rows.append(v)
            return rows
        except UnicodeDecodeError:
            continue
    return rows

def ensure_posix(s: str) -> str:
    return s.replace("\\", "/")

def build_chan_paths(sig_path_posix: str) -> Tuple[Path, str, Path]:
    """
    From exact CSV path (POSIX string), create:
      - base_rel_no_ext: relative path under extracted/ to the .flac w/o extension (includes split).
      - base_stem: file stem without extension.
      - base_dir_rel: relative directory under extracted/ for this file.
    """
    rel = Path(sig_path_posix)             # e.g., 'train/ma_speech/.../XYZ.flac'
    rel_no_ext = rel.with_suffix("")       # drop .flac
    base_stem = rel_no_ext.name            # '...W0001' or 'TRAIN_S_AUDI_0029_0001'
    base_dir_rel = rel_no_ext.parent       # 'train/ma_speech/...'
    return rel_no_ext, base_stem, base_dir_rel

def replacements_for_all_folders(sig_path_posix: str) -> List[str]:
    """
    For EVERY CSV row, try all MIRROR_FOLDERS by swapping the second segment
    (after split: train/|val/|test/) to each target folder. Also include the original path.
    """
    p = sig_path_posix.strip("/").split("/")
    candidates = [sig_path_posix]
    if p and p[0] in {"train", "val", "test"} and len(p) >= 2:
        for target in MIRROR_FOLDERS:
            q = p.copy()
            q[1] = target
            candidates.append("/".join(q))
    # de-dup while preserving order
    out, seen = [], set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def copy_preserving_under_extracted(src: Path) -> Tuple[Path, str]:
    """
    Copy src into dest_root/extracted/<relative-to-data_root>, crash-safe + resumable:
      - Skip if dst exists and same size as src.
      - Otherwise copy to dst.part and atomically replace.
    Returns (dst_path, status) where status in {"skipped_exists", "copied"}.
    """
    try:
        rel = src.resolve().relative_to(data_root.resolve())
    except Exception:
        rel = src.name  # fallback; should not happen if src under extracted

    dst = dest_extracted / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_size = src.stat().st_size

    if dst.exists():
        try:
            if dst.stat().st_size == src_size:
                return dst, "skipped_exists"
        except FileNotFoundError:
            pass

    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        if tmp.exists():
            tmp.unlink()
        shutil.copy2(src, tmp)
        if tmp.stat().st_size != src_size:
            raise IOError(f"Partial copy size mismatch: {tmp} vs {src}")
        os.replace(tmp, dst)  # atomic
        return dst, "copied"
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

def find_single_wav_candidates(base_dir_abs: Path, base_stem: str) -> List[Path]:
    """
    Return a list of plausible single-file .wav candidates for a basename that may include a trailing _W####.
    Tries, in order:
      1) <base_stem>.wav
      2) <base_stem_without__W\d+>.wav  (if applicable)
      3) Any *.wav in the folder that starts with the prefix before '_W' and does NOT contain '_CH'
    """
    candidates = []

    # 1) exact
    exact = base_dir_abs / f"{base_stem}.wav"
    if exact.exists():
        candidates.append(exact)

    # 2) remove trailing _W#### if present
    m = re.search(r"(.*)_W\d+$", base_stem)
    if m:
        noW = m.group(1)
        alt = base_dir_abs / f"{noW}.wav"
        if alt.exists() and alt not in candidates:
            candidates.append(alt)

        # 3) prefix scan: files starting with the prefix before _W and not containing _CH
        # (e.g., TRAIN_S_AUDI_0029_0001.wav)
        for p in base_dir_abs.glob(f"{noW}*.wav"):
            if "_CH" not in p.stem:
                if p not in candidates:
                    candidates.append(p)
    return candidates

def process():
    dest_logs = dest_root / "logs"
    dest_logs.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    total_skipped = 0

    for split in SPLIT_DIRS:
        split_dir = root / split
        if not split_dir.exists():
            continue

        csv_files = [p for p in split_dir.glob("*.csv") if CSV_NAME_PATTERN in p.name]
        if not csv_files:
            print(f"[{split}] No CSVs with '{CSV_NAME_PATTERN}' found.")
            continue

        for csv_path in csv_files:
            print(f"\n==> Processing CSV: {csv_path.relative_to(root)}")
            if COPY_CSVS_TOO:
                (dest_root / split).mkdir(parents=True, exist_ok=True)
                shutil.copy2(csv_path, dest_root / split / csv_path.name)

            rows = read_csv_rows(csv_path)
            print(f"   Rows: {len(rows)}")

            # Folder-wise counters for this CSV
            folder_copied = {k: 0 for k in MIRROR_FOLDERS}
            folder_skipped = {k: 0 for k in MIRROR_FOLDERS}

            # Small hint for first row
            if rows:
                first_rel = ensure_posix(rows[0])
                _, first_stem, first_dir_rel = build_chan_paths(first_rel)
                print(f"   Example dir under extracted/: {first_dir_rel}  | stem: {first_stem}")

            csv_copied = 0
            csv_skipped = 0
            csv_missing = 0

            for idx, base_value in enumerate(rows, 1):
                sig_path_posix = ensure_posix(base_value)

                # Try original + mirrored folder variants (always all four)
                for variant in replacements_for_all_folders(sig_path_posix):
                    rel_no_ext, base_stem, base_dir_rel = build_chan_paths(variant)
                    base_dir_abs = data_root / base_dir_rel  # e.g., D:\RealMAN\extracted\train\dp_speech\...

                    # Ensure destination directory exists even if everything is skipped
                    dest_base_dir = dest_extracted / base_dir_rel
                    dest_base_dir.mkdir(parents=True, exist_ok=True)

                    # which folder does this variant target? (second segment after split)
                    parts = variant.strip("/").split("/")
                    variant_folder = parts[1] if (len(parts) >= 2 and parts[0] in {"train","val","test"}) else None

                    # ---- DEBUG: show existence & counts for first few rows only ----
                    if idx <= DEBUG_FIRST_N_ROWS and variant_folder:
                        existing_ch = 0
                        single_exists = False
                        if base_dir_abs.exists():
                            if (base_dir_abs / f"{base_stem}_CH0.wav").exists():
                                for ch in range(32):
                                    if (base_dir_abs / f"{base_stem}_CH{ch}.wav").exists():
                                        existing_ch += 1
                            # single candidates quick probe
                            cands = find_single_wav_candidates(base_dir_abs, base_stem)
                            single_exists = len(cands) > 0
                        print(f"      [row {idx:>4}] variant folder={variant_folder:<14} "
                              f"exists={'Y' if base_dir_abs.exists() else 'N'} "
                              f"channels_found={existing_ch}/32 single={'Y' if single_exists else 'N'}")

                    if not base_dir_abs.exists():
                        # only mark 'no candidates' for the ORIGINAL path
                        if variant == sig_path_posix:
                            no_candidates.append((csv_path.name, base_value))
                        continue

                    # --- Handle both multi-channel and single-file cases ---
                    chan0 = base_dir_abs / f"{base_stem}_CH0.wav"
                    if chan0.exists():
                        # multi-channel: copy all 32 (log missing)
                        for ch in range(32):
                            chan_name = f"{base_stem}_CH{ch}.wav"
                            src = base_dir_abs / chan_name
                            if src.exists() and src.suffix.lower() == ".wav":
                                dst, status = copy_preserving_under_extracted(src)
                                if status == "skipped_exists":
                                    skipped_exists.append((csv_path.name, str(src), str(dst)))
                                    csv_skipped += 1
                                    total_skipped += 1
                                    if variant_folder in folder_skipped:
                                        folder_skipped[variant_folder] += 1
                                    if total_skipped % PROGRESS_EVERY == 0:
                                        print(f"   ...skipped {total_skipped} files so far")
                                else:
                                    copied.append((csv_path.name, str(src), str(dst)))
                                    csv_copied += 1
                                    total_copied += 1
                                    if variant_folder in folder_copied:
                                        folder_copied[variant_folder] += 1
                                    if total_copied % PROGRESS_EVERY == 0:
                                        print(f"   ...copied {total_copied} files so far")
                            else:
                                if variant == sig_path_posix:
                                    tried = [str(src)]
                                    missing_channels.append(
                                        (csv_path.name, str(base_dir_rel), base_stem, ch, ";".join(tried))
                                    )
                                    csv_missing += 1
                    else:
                        # single-file fallback with smarter matching
                        single_candidates = find_single_wav_candidates(base_dir_abs, base_stem)
                        if single_candidates:
                            for single in single_candidates:
                                dst, status = copy_preserving_under_extracted(single)
                                if status == "skipped_exists":
                                    skipped_exists.append((csv_path.name, str(single), str(dst)))
                                    csv_skipped += 1
                                    total_skipped += 1
                                    if variant_folder in folder_skipped:
                                        folder_skipped[variant_folder] += 1
                                    if total_skipped % PROGRESS_EVERY == 0:
                                        print(f"   ...skipped {total_skipped} files so far")
                                else:
                                    copied.append((csv_path.name, str(single), str(dst)))
                                    csv_copied += 1
                                    total_copied += 1
                                    if variant_folder in folder_copied:
                                        folder_copied[variant_folder] += 1
                                    if folder_copied.get(variant_folder, 0) == 1:
                                        print(f"   [+] First single-file copy from '{variant_folder}' for this CSV.")
                                    if total_copied % PROGRESS_EVERY == 0:
                                        print(f"   ...copied {total_copied} files so far")
                        else:
                            if variant == sig_path_posix:
                                no_candidates.append((csv_path.name, base_value))

                if idx % ROW_PROGRESS_EVERY == 0:
                    per_folder = " ".join([f"{k}=C{folder_copied[k]}/S{folder_skipped[k]}" for k in MIRROR_FOLDERS])
                    print(f"   Row {idx}/{len(rows)} processed... "
                          f"(csv copied: {csv_copied}, skipped: {csv_skipped}) [{per_folder}]")

            print(f"   Finished {csv_path.name}: copied={csv_copied}, skipped={csv_skipped}, missing_ch={csv_missing}")
            print("   Per-folder copied/skipped:",
                  " ".join([f"{k}={folder_copied[k]}/{folder_skipped[k]}" for k in MIRROR_FOLDERS]))
            print(f"   Totals so far: copied={total_copied}, skipped={total_skipped}")

    # write logs
    logs = dest_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    with (logs / "copied.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["csv", "src", "dst"])
        w.writerows(copied)

    with (logs / "skipped_exists.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["csv", "src", "dst"])
        w.writerows(skipped_exists)

    with (logs / "missing_channels.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["csv", "base_dir_rel", "base_stem", "channel", "tried_paths"])
        w.writerows(missing_channels)

    with (logs / "no_candidates_found.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["csv", "csv_filename_value"])
        w.writerows(no_candidates)

    print("\n=== SUMMARY ===")
    print(f"Copied files: {len(copied)}")
    print(f"Skipped (already present): {len(skipped_exists)}")
    print(f"Missing channel entries: {len(missing_channels)}")
    print(f"No candidates found: {len(no_candidates)}")
    print(f"Output at: {dest_root}")
    print(f"Logs at: {logs}")

if __name__ == "__main__":
    if not (root / DATA_DIR).exists():
        raise SystemExit(f"[ERROR] Data folder not found: {root / DATA_DIR}")
    process()
