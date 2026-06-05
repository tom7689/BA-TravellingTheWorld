import os
import shutil
from PIL import Image
from tqdm import tqdm

BASE_DIR = r"C:\Users\Work\Documents\zhaw\ba\ImagesAll2"

MIN_W = 400
MIN_H = 400

DRY_RUN = False  # zuerst lieber True testen

deleted = 0
checked = 0
errors = 0


def is_image_small(path):
    global errors
    try:
        with Image.open(path) as img:
            w, h = img.size
            return w < MIN_W or h < MIN_H
    except Exception as e:
        print(f"Error reading {path}: {e}")
        errors += 1
        return False


folders = [
    os.path.join(BASE_DIR, folder)
    for folder in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, folder))
]

pbar = tqdm(folders, desc="Checking folders", unit="folder")

for folder_path in pbar:
    checked += 1

    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        pbar.set_postfix(checked=checked, deleted=deleted, errors=errors)
        continue

    all_small = True

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)

        # sobald ein Bild groß genug ist, Ordner behalten
        if not is_image_small(img_path):
            all_small = False
            break

    if all_small:
        print(f"DELETE: {folder_path}")
        if not DRY_RUN:
            shutil.rmtree(folder_path)
        deleted += 1

    pbar.set_postfix(checked=checked, deleted=deleted, errors=errors)

print("\n--- SUMMARY ---")
print(f"Checked folders: {checked}")
print(f"Deleted folders: {deleted}")
print(f"Errors: {errors}")