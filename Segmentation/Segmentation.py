import os
import json
import time

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage

import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

# Input-Datei mit allen Bildern/Szenen
SCENES_JSON = r"Scenes\scenes.json"

# Ausgabeordner für Segmentierungs-JSONs und Segmentationskarten
OUT_JSON_DIR = r"Segmentation\output_segmentation\json"
OUT_SEGMAP_DIR = r"Segmentation\output_segmentation\segmaps"

# Mask2Former-Modell
MODEL_NAME = "facebook/mask2former-swin-large-ade-panoptic"

# Gerät für die Berechnung
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Postprocessing-Parameter für Mask2Former
POST_THRESHOLD = 0.5
POST_MASK_THRESHOLD = 0.5
POST_OVERLAP_THRESHOLD = 0.6

# Minimale Segmentgrösse relativ zur Bildfläche
MIN_AREA_RATIO = 0.001


# -------------------------------------------------
# BASIC HELPERS
# -------------------------------------------------

def ensure_dir(path: str):
    """Erstellt einen Ordner, falls er noch nicht existiert."""
    os.makedirs(path, exist_ok=True)


def load_scenes(path: str):
    """Lädt die Szenen-/Bildliste aus einer JSON-Datei."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------
# SEGMAP POSTPROCESSING
# -------------------------------------------------

def fill_invalid_ids_with_nearest(segmap: np.ndarray, valid_ids: set[int]):
    """
    Ersetzt alle Pixel mit ungültigen Segment-IDs durch das nächstgelegene gültige Segment.

    Ungültig bedeutet:
    Die ID kommt in der Segmentkarte vor, ist aber nicht in segments_info enthalten.

    Dadurch werden z. B. Void-Bereiche oder sonstige nicht zuordenbare Pixel gefüllt.
    """
    segmap = segmap.copy()

    invalid_mask = ~np.isin(segmap, list(valid_ids))

    if not invalid_mask.any():
        return segmap

    valid_mask = ~invalid_mask

    if not valid_mask.any():
        return segmap

    _, indices = ndimage.distance_transform_edt(
        invalid_mask,
        return_indices=True,
    )

    nearest_y = indices[0]
    nearest_x = indices[1]

    segmap[invalid_mask] = segmap[nearest_y[invalid_mask], nearest_x[invalid_mask]]

    return segmap


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------

def main():
    start_time = time.perf_counter()

    # Ausgabeordner vorbereiten
    ensure_dir(OUT_JSON_DIR)
    ensure_dir(OUT_SEGMAP_DIR)

    # Gerät ausgeben
    print("DEVICE:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # Modell laden
    print("Loading model...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    id2label = model.config.id2label

    # Szenen laden
    scenes = load_scenes(SCENES_JSON)
    print(f"Scenes: {len(scenes)}")

    try:
        for scene in tqdm(scenes):
            scene_id = scene["sceneId"]
            image_path = scene["path"]

            out_json_path = os.path.join(OUT_JSON_DIR, f"{scene_id}.json")

            # Bereits verarbeitete Bilder überspringen
            if os.path.exists(out_json_path):
                continue

            # Bild laden
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Skipping image {image_path}: {e}")
                continue

            # -------------------------------------------------
            # MASK2FORMER SEGMENTATION
            # -------------------------------------------------

            try:
                inputs = processor(images=image, return_tensors="pt").to(DEVICE)

                with torch.inference_mode():
                    if DEVICE == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)

                result = processor.post_process_panoptic_segmentation(
                    outputs,
                    target_sizes=[image.size[::-1]],
                    threshold=POST_THRESHOLD,
                    mask_threshold=POST_MASK_THRESHOLD,
                    overlap_mask_area_threshold=POST_OVERLAP_THRESHOLD,
                    label_ids_to_fuse=set(),
                )[0]

            except Exception as e:
                print(f"Segmentation failed for {image_path}: {e}")
                continue

            # Segmentkarte und Segmentinformationen auslesen
            seg_map = result["segmentation"].cpu().numpy()
            segments_info = result["segments_info"]

            # Ungültige IDs / Void-Bereiche mit nächstgelegenen gültigen Segmenten füllen
            valid_ids = {int(seg["id"]) for seg in segments_info}
            seg_map = fill_invalid_ids_with_nearest(seg_map, valid_ids)

            # Speicherformat je nach maximaler Segment-ID wählen
            max_id = int(seg_map.max())

            if max_id <= 255:
                seg_map = seg_map.astype(np.uint8)
            elif max_id <= 65535:
                seg_map = seg_map.astype(np.uint16)
            else:
                seg_map = seg_map.astype(np.uint32)

            # Segmentkarte speichern
            segmap_file = f"{scene_id}_segmap.npz"
            segmap_path = os.path.join(OUT_SEGMAP_DIR, segmap_file)

            np.savez_compressed(segmap_path, segmap=seg_map)

            # -------------------------------------------------
            # SEGMENT METADATA
            # -------------------------------------------------

            segments = []
            image_area = image.width * image.height

            for seg in segments_info:
                try:
                    segment_id = int(seg["id"])
                    category_id = int(seg["label_id"])
                    label = str(id2label[category_id]).strip().lower()
                    score = float(seg.get("score", 0.0))

                    # Maske für dieses Segment berechnen
                    mask = seg_map == segment_id
                    area = int(mask.sum())
                    area_ratio = area / image_area

                    # Zu kleine Segmente ignorieren
                    if area_ratio < MIN_AREA_RATIO:
                        continue

                    # Bounding Box berechnen
                    ys, xs = np.where(mask)

                    if len(xs) == 0 or len(ys) == 0:
                        continue

                    x1 = int(xs.min())
                    x2 = int(xs.max())
                    y1 = int(ys.min())
                    y2 = int(ys.max())

                    segments.append({
                        "segment_id": segment_id,
                        "category_id": category_id,
                        "label": label,
                        "area": area,
                        "area_ratio": area_ratio,
                        "score": score,
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": int(x2 - x1 + 1),
                            "height": int(y2 - y1 + 1),
                        }
                    })

                except Exception as e:
                    print(f"Skipping segment in {scene_id}: {e}")
                    continue

            # -------------------------------------------------
            # OUTPUT JSON
            # -------------------------------------------------

            record = {
                "image_id": scene_id,
                "image_path": image_path,
                "width": image.width,
                "height": image.height,
                "segmentation_map_npy": segmap_file,
                "postprocess": {
                    "threshold": POST_THRESHOLD,
                    "mask_threshold": POST_MASK_THRESHOLD,
                    "overlap_mask_area_threshold": POST_OVERLAP_THRESHOLD,
                    "min_area_ratio": MIN_AREA_RATIO,
                    "invalid_ids_filled": True,
                },
                "segments": segments,
            }

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

    except KeyboardInterrupt:
        print("\nStopped by user. Already processed files are kept.")

    # -------------------------------------------------
    # RUNTIME OUTPUT
    # -------------------------------------------------

    elapsed = time.perf_counter() - start_time

    print(f"⏱️ Total runtime: {elapsed:.2f} seconds")
    print(f"⏱️ Total runtime: {elapsed / 60:.2f} minutes")
    print(f"⏱️ Total runtime: {elapsed / 3600:.2f} hours")

    print("Done.")


if __name__ == "__main__":
    main()