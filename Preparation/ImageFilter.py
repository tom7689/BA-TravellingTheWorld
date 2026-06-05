import os
import shutil
from pathlib import Path

import torch
import open_clip
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

INPUT_DIR = r"C:\Users\Work\webcam_samples_all_6_each"
OUTPUT_DIR = r"C:\Users\Work\Documents\zhaw\ba\ImagesAll2Filtered"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
BATCH_SIZE = 32
BORDER = 0.10
DRY_RUN = False

# Night decision thresholds
NIGHT_STRICT = 0.80
NIGHT_LOOSE = 0.60
Q20_DARK = 25
Q20_TWILIGHT = 18
MED_TWILIGHT = 55

# Simple brightness shortcuts
BRIGHT_DAY_MEDIAN = 90   # if median > this => definitely day
DARK_NIGHT_MEDIAN = 20   # if median < this and q20 low => definitely night
DARK_NIGHT_Q20 = 10

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Model Setup
# ------------------------------------------------------------

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED
)
model = model.to(device).eval()

tokenizer = open_clip.get_tokenizer(MODEL_NAME)

TEXT_DAY = [
    "a photo taken during the day",
    "a daytime outdoor photo",
    "a bright daytime scene",
    "a photo taken at dawn",
    "a photo taken at dusk",
    "a photo taken at sunset",
    "a photo taken at twilight",
]

TEXT_NIGHT = [
    "a photo taken at night",
    "a nighttime outdoor photo",
    "a dark night scene",
    "a night landscape with city lights",
    "a night sky with clouds illuminated by city lights",
    "light pollution at night",
]

TEXT_ALL = TEXT_DAY + TEXT_NIGHT
DAY_IDX = list(range(len(TEXT_DAY)))
NIGHT_IDX = list(range(len(TEXT_DAY), len(TEXT_ALL)))

with torch.no_grad():
    tokens = tokenizer(TEXT_ALL).to(device)
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def collect_images(input_dir: str):
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fn in filenames:
            if Path(fn).suffix.lower() in VALID_EXTS:
                files.append(os.path.join(root, fn))
    return files


def crop_border_pil(img: Image.Image, border=0.10) -> Image.Image:
    w, h = img.size
    x1 = int(w * border)
    y1 = int(h * border)
    x2 = int(w * (1 - border))
    y2 = int(h * (1 - border))
    return img.crop((x1, y1, x2, y2))


def brightness_stats(image_path: str, border=0.10):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    h, w = img.shape[:2]
    y1 = int(h * border)
    y2 = int(h * (1 - border))
    x1 = int(w * border)
    x2 = int(w * (1 - border))
    img = img[y1:y2, x1:x2]

    if img.size == 0:
        return None, None

    # kleiner machen für schnellere Statistik
    if img.shape[1] > 512:
        scale = 512 / img.shape[1]
        new_w = 512
        new_h = max(1, int(img.shape[0] * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    median = float(np.median(img))
    q20 = float(np.quantile(img, 0.20))
    return median, q20


def load_preprocessed_tensor(image_path: str, border=0.10):
    img = Image.open(image_path).convert("RGB")
    img = crop_border_pil(img, border)
    return preprocess(img)


@torch.no_grad()
def get_day_night_scores_batch(image_paths, border=0.10):
    tensors = []
    valid_paths = []

    for path in image_paths:
        try:
            tensor = load_preprocessed_tensor(path, border)
            tensors.append(tensor)
            valid_paths.append(path)
        except Exception:
            continue

    if not tensors:
        return {}

    batch = torch.stack(tensors).to(device)

    img_features = model.encode_image(batch)
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    logits = (img_features @ text_features.T) * 100.0
    probs = logits.softmax(dim=-1)

    result = {}
    for i, path in enumerate(valid_paths):
        day_score = float(probs[i, DAY_IDX].sum().item())
        night_score = float(probs[i, NIGHT_IDX].sum().item())
        result[path] = (day_score, night_score)

    return result


def move_file_preserve_structure(src_path: str, input_dir: str, out_dir: str, dry_run=False):
    relative_file = os.path.relpath(src_path, input_dir)
    dst_path = os.path.join(out_dir, relative_file)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if dry_run:
        return dst_path

    shutil.move(src_path, dst_path)
    return dst_path

# ------------------------------------------------------------
# Main Logic
# ------------------------------------------------------------

def filter_folder_batched(
    input_dir: str,
    out_night_dir: str,
    batch_size=32,
    border=0.10,
    dry_run=False,
):
    os.makedirs(out_night_dir, exist_ok=True)

    files = collect_images(input_dir)
    total = len(files)

    print(f"Found {total} images")
    print(f"Using device: {device}")

    moved = 0
    errors = 0
    clip_checked = 0

    # Step 1: brightness pre-check
    definitely_day = set()
    definitely_night = set()
    need_clip = []

    pbar = tqdm(files, desc="Brightness scan", unit="img")

    for path in pbar:
        try:
            median, q20 = brightness_stats(path, border)

            if median is None:
                errors += 1
                pbar.set_postfix(moved=moved, clip=len(need_clip), errors=errors)
                continue

            if median > BRIGHT_DAY_MEDIAN:
                definitely_day.add(path)
            elif median < DARK_NIGHT_MEDIAN and q20 < DARK_NIGHT_Q20:
                definitely_night.add(path)
            else:
                need_clip.append((path, median, q20))

            pbar.set_postfix(
                bright_day=len(definitely_day),
                bright_night=len(definitely_night),
                clip=len(need_clip),
                errors=errors,
            )
        except Exception:
            errors += 1
            pbar.set_postfix(
                bright_day=len(definitely_day),
                bright_night=len(definitely_night),
                clip=len(need_clip),
                errors=errors,
            )

    # Step 2: move definitely night
    if definitely_night:
        pbar_move = tqdm(definitely_night, desc="Moving obvious night", unit="img")
        for path in pbar_move:
            try:
                move_file_preserve_structure(path, input_dir, out_night_dir, dry_run=dry_run)
                moved += 1
                pbar_move.set_postfix(moved=moved, errors=errors)
            except Exception:
                errors += 1
                pbar_move.set_postfix(moved=moved, errors=errors)

    # Step 3: CLIP on uncertain images in batches
    pbar_clip = tqdm(
        range(0, len(need_clip), batch_size),
        desc="CLIP batches",
        unit="batch"
    )

    for start in pbar_clip:
        chunk = need_clip[start:start + batch_size]
        chunk_paths = [x[0] for x in chunk]

        try:
            scores = get_day_night_scores_batch(chunk_paths, border=border)
            clip_checked += len(chunk_paths)

            for path, median, q20 in chunk:
                if path not in scores:
                    errors += 1
                    continue

                _, night_score = scores[path]

                is_night = False

                if night_score >= NIGHT_STRICT:
                    is_night = True
                elif night_score >= NIGHT_LOOSE and q20 < Q20_DARK:
                    is_night = True
                elif q20 < Q20_TWILIGHT and median < MED_TWILIGHT:
                    is_night = True

                if is_night:
                    try:
                        move_file_preserve_structure(
                            path, input_dir, out_night_dir, dry_run=dry_run
                        )
                        moved += 1
                    except Exception:
                        errors += 1

            pbar_clip.set_postfix(
                moved=moved,
                clip_checked=clip_checked,
                errors=errors,
            )

        except Exception:
            errors += len(chunk_paths)
            pbar_clip.set_postfix(
                moved=moved,
                clip_checked=clip_checked,
                errors=errors,
            )

    print("\nDone")
    print(f"Total images       : {total}")
    print(f"Moved night images : {moved}")
    print(f"CLIP checked       : {clip_checked}")
    print(f"Errors             : {errors}")
    print(f"Definitely day     : {len(definitely_day)}")
    print(f"Definitely night   : {len(definitely_night)}")


if __name__ == "__main__":
    filter_folder_batched(
        input_dir=INPUT_DIR,
        out_night_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        border=BORDER,
        dry_run=DRY_RUN,
    )