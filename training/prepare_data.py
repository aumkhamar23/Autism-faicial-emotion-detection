"""
Phase 1 — Data preparation pipeline.

What this script does:
  1. Scans both raw datasets (FER2013 + CK+) and collects all image paths per emotion.
  2. Normalises class names to the unified EMOTIONS list.
  3. Balances classes:
       - Caps dominant classes at MAX_PER_CLASS (random under-sample).
       - Augments minority classes up to MIN_PER_CLASS (flip, rotate, brightness, noise).
  4. Splits balanced set into train / val / test (70/15/15).
  5. Resizes every image to IMG_SIZE × IMG_SIZE RGB and writes to data/processed/.

Run:
    python training/prepare_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from collections import defaultdict

from training.config import (
    DATA_RAW, DATA_PROCESSED, EMOTIONS, NUM_CLASSES,
    IMG_SIZE, MAX_PER_CLASS, MIN_PER_CLASS,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED, DATASET_SOURCES,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def augment_image(img: np.ndarray) -> np.ndarray:
    """Return one randomly augmented copy of img (H×W×3 uint8)."""
    aug = img.copy()

    # Horizontal flip (50 %)
    if random.random() < 0.5:
        aug = cv2.flip(aug, 1)

    # Rotation ±12°
    angle = random.uniform(-12, 12)
    h, w = aug.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Brightness / contrast jitter
    alpha = random.uniform(0.80, 1.20)   # contrast
    beta  = random.randint(-20, 20)       # brightness
    aug = np.clip(aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Gaussian noise
    if random.random() < 0.4:
        noise = np.random.normal(0, 8, aug.shape).astype(np.float32)
        aug = np.clip(aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return aug


# ---------------------------------------------------------------------------
# Step 1 — collect all raw image paths grouped by unified emotion label
# ---------------------------------------------------------------------------

def collect_raw_paths() -> dict[str, list[Path]]:
    """Return {emotion: [path, ...]} across all configured sources."""
    pools: dict[str, list[Path]] = defaultdict(list)

    for ds_name, cfg in DATASET_SOURCES.items():
        root: Path = cfg['root']
        splits      = cfg['splits']
        label_map   = cfg['label_map']

        if not root.exists():
            print(f"  [WARN] {ds_name}: root not found → {root}")
            continue

        search_roots = (
            [root / s for s in splits] if splits else [root]
        )

        for sr in search_roots:
            if not sr.exists():
                print(f"  [WARN] {ds_name}: split dir not found → {sr}")
                continue
            for emotion_dir in sorted(sr.iterdir()):
                if not emotion_dir.is_dir():
                    continue
                raw_label = emotion_dir.name.lower()
                unified   = label_map.get(raw_label)
                if unified is None:
                    continue          # skipped class (e.g. contempt)
                if unified not in EMOTIONS:
                    print(f"  [WARN] {ds_name}: '{raw_label}' mapped to '{unified}' not in EMOTIONS — skipping")
                    continue
                imgs = [p for p in emotion_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
                pools[unified].extend(imgs)
                print(f"  {ds_name}/{raw_label} -> {unified}: {len(imgs)} images")

    return pools


# ---------------------------------------------------------------------------
# Step 2 — balance classes (cap + augment)
# ---------------------------------------------------------------------------

def balance_pools(pools: dict[str, list[Path]]) -> dict[str, list]:
    """
    Returns {emotion: [(path_or_ndarray, is_augmented), ...]} where each list
    has between MIN_PER_CLASS and MAX_PER_CLASS items.
    """
    balanced: dict[str, list] = {}

    for emotion in EMOTIONS:
        paths = pools.get(emotion, [])
        random.shuffle(paths)

        # Under-sample dominant classes
        if len(paths) > MAX_PER_CLASS:
            paths = paths[:MAX_PER_CLASS]

        items = [(p, False) for p in paths]

        # Augment minority classes
        if len(items) < MIN_PER_CLASS:
            needed = MIN_PER_CLASS - len(items)
            aug_pool = paths * ((needed // max(len(paths), 1)) + 2)
            random.shuffle(aug_pool)
            for src_path in aug_pool[:needed]:
                img = _load_and_resize(src_path)
                if img is not None:
                    items.append((augment_image(img), True))

        balanced[emotion] = items
        n_real = sum(1 for _, aug in items if not aug)
        n_aug  = sum(1 for _, aug in items if aug)
        print(f"  {emotion:12s}: {n_real} real + {n_aug} augmented = {len(items)} total")

    return balanced


# ---------------------------------------------------------------------------
# Step 3 — train/val/test split
# ---------------------------------------------------------------------------

def split_items(items: list) -> tuple[list, list, list]:
    random.shuffle(items)
    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_and_resize(src) -> np.ndarray | None:
    """Load from path or ndarray, resize to IMG_SIZE×IMG_SIZE RGB."""
    if isinstance(src, np.ndarray):
        img = src
    else:
        img = cv2.imread(str(src))
        if img is None:
            return None
    if len(img.shape) == 2:                        # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                        # BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def _save(item, dest_path: Path) -> bool:
    src, is_aug = item
    img = _load_and_resize(src)
    if img is None:
        return False
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest_path), img)
    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("\n=== Step 1: Collecting raw image paths ===")
    pools = collect_raw_paths()

    print("\nRaw totals before balancing:")
    for em in EMOTIONS:
        print(f"  {em:12s}: {len(pools.get(em, []))}")

    print(f"\n=== Step 2: Balancing (cap={MAX_PER_CLASS}, min={MIN_PER_CLASS}) ===")
    balanced = balance_pools(pools)

    print("\n=== Step 3: Splitting into train / val / test ===")
    splits = {'train': [], 'val': [], 'test': []}
    for emotion, items in balanced.items():
        tr, va, te = split_items(items)
        splits['train'].append((emotion, tr))
        splits['val'].append((emotion, va))
        splits['test'].append((emotion, te))
        print(f"  {emotion:12s}: train={len(tr)}  val={len(va)}  test={len(te)}")

    print("\n=== Step 4: Writing processed images ===")
    if DATA_PROCESSED.exists():
        shutil.rmtree(DATA_PROCESSED)

    totals = defaultdict(int)
    for split_name, emotion_items in splits.items():
        for emotion, items in emotion_items:
            dest_dir = DATA_PROCESSED / split_name / emotion
            dest_dir.mkdir(parents=True, exist_ok=True)
            saved = 0
            for idx, item in enumerate(items):
                dest_file = dest_dir / f"{emotion}_{split_name}_{idx:05d}.jpg"
                if _save(item, dest_file):
                    saved += 1
            totals[split_name] += saved
        print(f"  {split_name}: done")

    print("\n=== Final dataset summary ===")
    for split_name in ['train', 'val', 'test']:
        print(f"\n  [{split_name}]")
        split_total = 0
        for emotion in EMOTIONS:
            d = DATA_PROCESSED / split_name / emotion
            count = len(list(d.glob('*.jpg'))) if d.exists() else 0
            print(f"    {emotion:12s}: {count}")
            split_total += count
        print(f"    {'TOTAL':12s}: {split_total}")

    print("\nData preparation complete.")
    print(f"Processed data saved to: {DATA_PROCESSED}")


if __name__ == '__main__':
    main()
