import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import ndimage

import torch
from transformers import Sam3Model, Sam3Processor


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

# Input from Mask2Former.
# The script assumes that every JSON has this fixed structure:
# image_id, image_path, segmentation_map_npy and segments.
SEGMENT_JSON_DIR = Path(r"Segmentation\output_segmentation\json")
SEGMAP_DIR = Path(r"Segmentation\output_segmentation\segmaps")

# Output after SAM3 correction.
PIPELINE_OUT_DIR = Path(r"Segmentation\output_sam3_correction")
OUT_JSON_DIR = PIPELINE_OUT_DIR / "json"
OUT_SEGMAP_DIR = PIPELINE_OUT_DIR / "segmaps"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If True, already processed JSON files are skipped.
RESUME = True

# Use an integer for testing, for example 20. Use None for the full dataset.
MAX_FILES: Optional[int] = None

ERROR_LOG_NAME = "sam3_correction_errors.json"


# -------------------------------------------------
# SAM3 CORRECTION SETTINGS
# -------------------------------------------------

# Only these Mask2Former labels are sent to SAM3.
# The new SAM3 children keep the same label as the parent.
REFINABLE_LABELS = {
    "building",
    "house",
}

# Padding around the parent segment before cropping the image for SAM3.
CROP_PADDING = 25

# SAM3 post-processing thresholds.
SAM3_THRESHOLD = 0.35
SAM3_MASK_THRESHOLD = 0.5

# Very small parent segments are copied directly without SAM3.
MIN_PARENT_AREA = 3000

# Very small SAM3 child masks are ignored.
MIN_CHILD_AREA = 800

# The accepted SAM3 mask must mostly lie inside the original Mask2Former parent.
MIN_PARENT_OVERLAP_RATIO = 0.55

# Clip SAM3 child masks to the original parent segment.
# This prevents SAM3 from stealing pixels from neighbouring regions.
CLIP_CHILDREN_TO_PARENT = True

# If SAM3 finds no usable children, the original Mask2Former segment is kept.
KEEP_ORIGINAL_IF_NO_CHILDREN = True

# Checks whether the final segmap still has holes or JSON/segmap ID mismatches.
DEBUG_VOID_CHECK = True


# -------------------------------------------------
# FINAL CLEANUP SETTINGS
# -------------------------------------------------

# Remove small disconnected islands after SAM3.
FINAL_REMOVE_TINY_COMPONENTS = True
FINAL_SMALL_COMPONENT_AREA = 800
FINAL_SMALL_COMPONENT_RATIO = 0.08
FINAL_CLEAN_CONNECTIVITY = 1
FINAL_CLEAN_ITERATIONS = 2

# Fill remaining 0 pixels with the nearest valid segment ID.
FINAL_FILL_ZERO_VOIDS = True

# Optional heuristic filter for suspicious building/house fragments.
FILTER_SUSPICIOUS_LABEL_SEGMENTS = True
SUSPICIOUS_LABEL_FILTERS = {
    "building": {
        "min_area": 12000,
        "min_height": 60,
        "min_fill_ratio": 0.20,
        "max_aspect_ratio": 5.5,
        "min_bad_conditions": 2,
    },
    "house": {
        "min_area": 5000,
        "min_height": 45,
        "min_fill_ratio": 0.22,
        "max_aspect_ratio": 5.0,
        "min_bad_conditions": 2,
    },
}


# -------------------------------------------------
# FILE HELPERS
# -------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_segmap(path: Path) -> np.ndarray:
    # Fixed input format: compressed NPZ with key "segmap".
    with np.load(path, allow_pickle=False) as data:
        return data["segmap"]


def save_segmap(path: Path, segmap: np.ndarray) -> None:
    ensure_dir(path.parent)

    max_id = int(segmap.max())
    if max_id <= 255:
        segmap = segmap.astype(np.uint8)
    elif max_id <= 65535:
        segmap = segmap.astype(np.uint16)
    else:
        segmap = segmap.astype(np.uint32)

    np.savez_compressed(path, segmap=segmap)


def normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def load_input(json_path: Path):
    """
    Load one Mask2Former result.

    This function is intentionally simple because the input JSON format is fixed:
    - data["image_id"]
    - data["image_path"]
    - data["segmentation_map_npy"]
    - data["segments"]
    """
    data = load_json(json_path)

    image_id = data["image_id"]
    image_path = Path(data["image_path"])
    segmap_path = SEGMAP_DIR / data["segmentation_map_npy"]
    segments = data["segments"]

    image = Image.open(image_path).convert("RGB")
    old_segmap = load_segmap(segmap_path)

    return data, image_id, image, old_segmap, segments, segmap_path


# -------------------------------------------------
# MASK AND SEGMENT HELPERS
# -------------------------------------------------

def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return x1, y1, x2, y2


def pad_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    padding: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    return x1, y1, x2, y2


def segment_record_from_mask(
    segment_id: int,
    label: str,
    mask: np.ndarray,
    score: float,
    parent_segment_id: Optional[int],
    parent_label: Optional[str],
    sam3_prompt: Optional[str],
    source: str,
) -> Optional[Dict[str, Any]]:
    """Create a JSON segment record from a binary mask."""
    area = int(mask.sum())
    if area <= 0:
        return None

    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    return {
        "segment_id": int(segment_id),
        "category_id": -1,
        "label": normalize_text(label),
        "area": area,
        "score": float(score),
        "was_fused": False,
        "source": source,
        "parent_segment_id": parent_segment_id,
        "parent_label": parent_label,
        "sam3_prompt": sam3_prompt,
        "bbox": {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2 - 1),
            "y2": int(y2 - 1),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
        },
    }


def refresh_segment_records_from_segmap(
    segmap: np.ndarray,
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Recompute area and bbox after all overlap removal and cleanup steps.
    Segments that no longer exist in the final segmap are removed from the JSON.
    """
    refreshed: List[Dict[str, Any]] = []

    for segment in segments:
        sid = int(segment["segment_id"])
        mask = segmap == sid
        area = int(mask.sum())

        if area <= 0:
            continue

        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox

        updated = dict(segment)
        updated["area"] = area
        updated["bbox"] = {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2 - 1),
            "y2": int(y2 - 1),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
        }
        refreshed.append(updated)

    return refreshed


# -------------------------------------------------
# SAFETY FILL
# -------------------------------------------------

def fill_remaining_old_pixels(
    old_segmap: np.ndarray,
    new_segmap: np.ndarray,
    new_segments: List[Dict[str, Any]],
    old_segments_by_id: Dict[int, Dict[str, Any]],
    next_segment_id: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]], int, int]:
    """
    Assign remaining old Mask2Former pixels to the nearest existing new segment.
    This avoids creating extra 'unknown' safety-fill segments.
    """
    old_valid = old_segmap > 0
    missing_mask = old_valid & (new_segmap == 0)
    filled_pixels = int(missing_mask.sum())

    if filled_pixels == 0:
        return new_segmap, new_segments, next_segment_id, 0

    valid_mask = new_segmap > 0

    # If there are already valid new segments, assign every missing pixel
    # to the nearest valid segment ID.
    if valid_mask.any():
        _, indices = ndimage.distance_transform_edt(
            ~valid_mask,
            return_indices=True,
        )

        nearest_y = indices[0]
        nearest_x = indices[1]

        new_segmap[missing_mask] = new_segmap[
            nearest_y[missing_mask],
            nearest_x[missing_mask]
        ]

        return new_segmap, new_segments, next_segment_id, filled_pixels

    # Fallback only if the image has no valid new segment at all.
    # This should normally not happen.
    missing_old_ids = [
        int(x) for x in np.unique(old_segmap[missing_mask])
        if int(x) > 0
    ]

    for old_id in missing_old_ids:
        mask = missing_mask & (old_segmap == old_id)
        if int(mask.sum()) <= 0:
            continue

        old_segment = old_segments_by_id.get(old_id, {})
        label = normalize_text(old_segment.get("label", "unknown"))
        score = float(old_segment.get("score", 1.0))

        new_segmap[mask] = next_segment_id

        record = segment_record_from_mask(
            segment_id=next_segment_id,
            label=label,
            mask=mask,
            score=score,
            parent_segment_id=old_id,
            parent_label=label,
            sam3_prompt=None,
            source="mask2former_safety_fill_fallback",
        )

        if record:
            new_segments.append(record)
            next_segment_id += 1

    return new_segmap, new_segments, next_segment_id, filled_pixels


# -------------------------------------------------
# FINAL SEGMAP CLEANUP
# -------------------------------------------------

def get_structure(connectivity: int = 1) -> np.ndarray:
    if connectivity == 2:
        return np.ones((3, 3), dtype=np.uint8)

    return np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )


def absorb_remainder_into_children(
    new_segmap: np.ndarray,
    parent_mask: np.ndarray,
    child_ids: List[int],
) -> np.ndarray:
    """
    Assign the remaining parent pixels to the nearest SAM3 child segment.
    This prevents old building/house parent segments from staying as ring shapes.
    """
    if not child_ids:
        return new_segmap

    child_seed_mask = np.isin(new_segmap, child_ids) & parent_mask
    if not child_seed_mask.any():
        return new_segmap

    remainder_mask = parent_mask & (~child_seed_mask)
    if not remainder_mask.any():
        return new_segmap

    _, indices = ndimage.distance_transform_edt(
        ~child_seed_mask,
        return_indices=True,
    )

    nearest_y = indices[0]
    nearest_x = indices[1]

    ys, xs = np.where(remainder_mask)
    nearest_ids = new_segmap[nearest_y[ys, xs], nearest_x[ys, xs]]

    valid = np.isin(nearest_ids, child_ids)
    new_segmap[ys[valid], xs[valid]] = nearest_ids[valid]

    return new_segmap


def remove_small_disconnected_components_per_segment(
    segmap: np.ndarray,
    min_area: int,
    min_largest_ratio: float,
    connectivity: int,
    iterations: int,
) -> np.ndarray:
    """
    Remove small disconnected islands for each segment ID by merging them into
    the most common neighbouring segment.
    """
    segmap = segmap.copy()
    structure = get_structure(connectivity)

    for _ in range(max(1, iterations)):
        changed_pixels = 0

        for sid in [int(x) for x in np.unique(segmap) if int(x) > 0]:
            mask = segmap == sid
            labeled, num = ndimage.label(mask, structure=structure)

            if num <= 1:
                continue

            component_sizes = np.bincount(labeled.ravel())
            component_sizes[0] = 0

            largest_component = int(component_sizes.argmax())
            largest_area = int(component_sizes[largest_component])

            if largest_area <= 0:
                continue

            for comp_id in range(1, num + 1):
                if comp_id == largest_component:
                    continue

                comp_area = int(component_sizes[comp_id])
                remove_by_abs = comp_area < min_area
                remove_by_ratio = comp_area < int(largest_area * min_largest_ratio)

                if not (remove_by_abs or remove_by_ratio):
                    continue

                comp_mask = labeled == comp_id

                dilated = ndimage.binary_dilation(
                    comp_mask,
                    structure=structure,
                    iterations=1,
                )
                border = dilated & (~comp_mask)

                neighbor_ids = segmap[border]
                neighbor_ids = neighbor_ids[(neighbor_ids > 0) & (neighbor_ids != sid)]

                if neighbor_ids.size == 0:
                    continue

                fill_id = int(np.bincount(neighbor_ids).argmax())
                segmap[comp_mask] = fill_id
                changed_pixels += comp_area

        if changed_pixels == 0:
            break

    return segmap


def fill_zero_voids_with_nearest(segmap: np.ndarray) -> np.ndarray:
    """Fill remaining 0 pixels with the nearest valid segment ID."""
    segmap = segmap.copy()

    void_mask = segmap == 0
    if not void_mask.any():
        return segmap

    valid_mask = segmap > 0
    if not valid_mask.any():
        return segmap

    _, indices = ndimage.distance_transform_edt(
        void_mask,
        return_indices=True,
    )

    nearest_y = indices[0]
    nearest_x = indices[1]
    segmap[void_mask] = segmap[nearest_y[void_mask], nearest_x[void_mask]]

    return segmap


def final_clean_sam3_segmap(segmap: np.ndarray) -> np.ndarray:
    """Run final cleanup before segment areas and bounding boxes are refreshed."""
    if FINAL_REMOVE_TINY_COMPONENTS:
        segmap = remove_small_disconnected_components_per_segment(
            segmap=segmap,
            min_area=FINAL_SMALL_COMPONENT_AREA,
            min_largest_ratio=FINAL_SMALL_COMPONENT_RATIO,
            connectivity=FINAL_CLEAN_CONNECTIVITY,
            iterations=FINAL_CLEAN_ITERATIONS,
        )

    if FINAL_FILL_ZERO_VOIDS:
        segmap = fill_zero_voids_with_nearest(segmap)

    return segmap


# -------------------------------------------------
# SUSPICIOUS BUILDING/HOUSE FILTER
# -------------------------------------------------

def bbox_stats_from_mask(mask: np.ndarray) -> Optional[Dict[str, float]]:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    width = int(x2 - x1)
    height = int(y2 - y1)

    if width <= 0 or height <= 0:
        return None

    area = int(mask.sum())
    bbox_area = int(width * height)

    return {
        "width": width,
        "height": height,
        "bbox_area": bbox_area,
        "fill_ratio": area / float(bbox_area),
        "aspect_ratio": max(width / float(height), height / float(width)),
        "area": area,
    }


def merge_mask_into_neighbors(
    segmap: np.ndarray,
    remove_mask: np.ndarray,
    remove_sid: int,
) -> np.ndarray:
    """Replace a removed segment area with the most common neighbouring ID."""
    segmap = segmap.copy()

    if not remove_mask.any():
        return segmap

    structure = get_structure(1)
    dilated = ndimage.binary_dilation(remove_mask, structure=structure, iterations=1)
    border = dilated & (~remove_mask)

    neighbor_ids = segmap[border]
    neighbor_ids = neighbor_ids[(neighbor_ids > 0) & (neighbor_ids != remove_sid)]

    if neighbor_ids.size == 0:
        return segmap

    fill_id = int(np.bincount(neighbor_ids).argmax())
    segmap[remove_mask] = fill_id

    return segmap


def remove_suspicious_label_segments(
    segmap: np.ndarray,
    segments: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Remove suspicious building/house fragments based on simple shape rules.
    This is a heuristic cleanup step and can be disabled in the config.
    """
    if not FILTER_SUSPICIOUS_LABEL_SEGMENTS:
        return segmap

    segmap = segmap.copy()

    for _ in range(2):
        changed = False

        for segment in segments:
            sid = int(segment.get("segment_id", -1))
            label = normalize_text(segment.get("label", ""))

            if sid <= 0 or label not in SUSPICIOUS_LABEL_FILTERS:
                continue

            mask = segmap == sid
            if not mask.any():
                continue

            stats = bbox_stats_from_mask(mask)
            if stats is None:
                continue

            cfg = SUSPICIOUS_LABEL_FILTERS[label]

            bad_conditions = 0
            if stats["area"] < int(cfg["min_area"]):
                bad_conditions += 1
            if stats["height"] < int(cfg["min_height"]):
                bad_conditions += 1
            if stats["fill_ratio"] < float(cfg["min_fill_ratio"]):
                bad_conditions += 1
            if stats["aspect_ratio"] > float(cfg["max_aspect_ratio"]):
                bad_conditions += 1

            if bad_conditions < int(cfg["min_bad_conditions"]):
                continue

            segmap = merge_mask_into_neighbors(
                segmap=segmap,
                remove_mask=mask,
                remove_sid=sid,
            )
            changed = True

        if not changed:
            break

    return segmap


# -------------------------------------------------
# SAM3
# -------------------------------------------------

def load_sam3():
    print(f"Loading SAM3 on {DEVICE}...")

    if DEVICE == "cuda":
        model = Sam3Model.from_pretrained(
            "facebook/sam3",
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)
    else:
        model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)

    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()

    return model, processor


def run_sam3_on_crop(
    crop: Image.Image,
    prompt: str,
    model,
    processor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run SAM3 on one cropped image region and return masks and scores."""
    inputs = processor(
        images=crop,
        text=prompt,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=SAM3_THRESHOLD,
        mask_threshold=SAM3_MASK_THRESHOLD,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = result["masks"]
    scores = result["scores"]

    if masks is None or len(masks) == 0:
        return np.empty((0, crop.height, crop.width), dtype=bool), np.empty((0,), dtype=float)

    masks_np = masks.detach().float().cpu().numpy() if isinstance(masks, torch.Tensor) else np.array(masks)
    scores_np = scores.detach().float().cpu().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)

    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0, :, :]

    return masks_np, scores_np


# -------------------------------------------------
# SAM3 CANDIDATES
# -------------------------------------------------

def collect_sam_candidates(
    crop: Image.Image,
    parent_crop_mask: np.ndarray,
    full_shape: Tuple[int, int],
    crop_offset: Tuple[int, int],
    parent_label: str,
    model,
    processor,
) -> List[Dict[str, Any]]:
    """
    Run SAM3 with the parent label as prompt and collect valid child masks.
    Example: parent label "building" -> SAM3 prompt "building".
    """
    height, width = full_shape
    x1, y1 = crop_offset
    prompt = parent_label

    candidates: List[Dict[str, Any]] = []

    try:
        masks_np, scores_np = run_sam3_on_crop(
            crop=crop,
            prompt=prompt,
            model=model,
            processor=processor,
        )
    except Exception as e:
        print(f"[WARN] SAM3 failed, parent={parent_label}, prompt={prompt}: {e}")
        return candidates

    for i, sam_mask_crop in enumerate(masks_np):
        sam_mask_crop = sam_mask_crop > 0.5

        sam_area = int(sam_mask_crop.sum())
        if sam_area <= 0:
            continue

        if CLIP_CHILDREN_TO_PARENT:
            corrected_crop_mask = sam_mask_crop & parent_crop_mask
        else:
            corrected_crop_mask = sam_mask_crop

        corrected_area = int(corrected_crop_mask.sum())
        if corrected_area < MIN_CHILD_AREA:
            continue

        overlap_ratio = corrected_area / float(sam_area)
        if overlap_ratio < MIN_PARENT_OVERLAP_RATIO:
            continue

        full_mask = np.zeros((height, width), dtype=bool)
        crop_h, crop_w = corrected_crop_mask.shape
        full_mask[y1:y1 + crop_h, x1:x1 + crop_w] = corrected_crop_mask

        score = float(scores_np[i]) if len(scores_np) > i else 0.0

        candidates.append({
            "mask": full_mask,
            "score": score,
            "prompt": prompt,
            "label": parent_label,
            "source": "sam3_same_label_child",
            "area": corrected_area,
        })

    return candidates


def accept_non_overlapping_candidates(
    candidates: List[Dict[str, Any]],
    parent_mask: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Accept candidate masks in score/area order and remove overlaps.
    Every accepted child remains inside the original parent segment.
    """
    candidates = sorted(
        candidates,
        key=lambda item: (item["score"], item["area"]),
        reverse=True,
    )

    accepted: List[Dict[str, Any]] = []
    occupied_parent_pixels = np.zeros_like(parent_mask, dtype=bool)

    for candidate in candidates:
        child_mask = candidate["mask"]

        if CLIP_CHILDREN_TO_PARENT:
            child_mask = child_mask & parent_mask

        child_mask = child_mask & (~occupied_parent_pixels)

        child_area = int(child_mask.sum())
        if child_area < MIN_CHILD_AREA:
            continue

        accepted.append({
            "mask": child_mask,
            "score": candidate["score"],
            "prompt": candidate["prompt"],
            "label": candidate["label"],
            "source": candidate["source"],
            "area": child_area,
        })

        occupied_parent_pixels |= child_mask

    return accepted


# -------------------------------------------------
# DEBUG CHECK
# -------------------------------------------------

def debug_check_segmap(
    json_path: Path,
    old_segmap: np.ndarray,
    new_segmap: np.ndarray,
    new_segments: List[Dict[str, Any]],
) -> None:
    if not DEBUG_VOID_CHECK:
        return

    old_valid = old_segmap > 0
    new_void_inside_old = old_valid & (new_segmap == 0)
    void_pixels = int(new_void_inside_old.sum())

    used_ids = set(int(x) for x in np.unique(new_segmap)) - {0}
    json_ids = set(int(s["segment_id"]) for s in new_segments)

    missing_json = sorted(used_ids - json_ids)
    missing_segmap = sorted(json_ids - used_ids)

    if void_pixels > 0:
        raise RuntimeError(
            f"SAFETY CHECK FAILED for {json_path.name}: "
            f"{void_pixels} pixels are void inside the old segmentation."
        )

    if missing_json:
        print(f"[WARN] {json_path.name}: segmap IDs without JSON entries: {missing_json}")

    if missing_segmap:
        print(f"[WARN] {json_path.name}: JSON IDs not present in segmap: {missing_segmap}")


# -------------------------------------------------
# PROCESS ONE IMAGE
# -------------------------------------------------

def copy_parent_segment(
    new_segmap: np.ndarray,
    new_segments: List[Dict[str, Any]],
    parent_segment: Dict[str, Any],
    parent_mask: np.ndarray,
    parent_id: int,
    parent_label: str,
    next_segment_id: int,
    source: str,
) -> int:
    """Copy an unchanged Mask2Former parent segment to the new output."""
    new_segmap[parent_mask] = next_segment_id

    record = segment_record_from_mask(
        segment_id=next_segment_id,
        label=parent_label,
        mask=parent_mask,
        score=float(parent_segment.get("score", 1.0)),
        parent_segment_id=parent_id,
        parent_label=parent_label,
        sam3_prompt=None,
        source=source,
    )

    if record:
        new_segments.append(record)
        next_segment_id += 1

    return next_segment_id


def process_one_json(
    json_path: Path,
    model,
    processor,
) -> None:
    data, image_id, image, old_segmap, old_segments, segmap_path = load_input(json_path)
    width, height = image.size

    if old_segmap.shape != (height, width):
        raise ValueError(
            f"Segmap/image size mismatch: segmap={old_segmap.shape}, image={(height, width)}"
        )

    old_segments_by_id = {
        int(segment["segment_id"]): segment
        for segment in old_segments
    }

    new_segmap = np.zeros((height, width), dtype=np.int32)
    new_segments: List[Dict[str, Any]] = []
    next_segment_id = 1

    for parent_segment in old_segments:
        parent_id = int(parent_segment["segment_id"])
        parent_label = normalize_text(parent_segment["label"])
        parent_mask = old_segmap == parent_id
        parent_area = int(parent_mask.sum())

        # Small segments and non-refinable labels are copied unchanged.
        if parent_area < MIN_PARENT_AREA:
            next_segment_id = copy_parent_segment(
                new_segmap=new_segmap,
                new_segments=new_segments,
                parent_segment=parent_segment,
                parent_mask=parent_mask,
                parent_id=parent_id,
                parent_label=parent_label,
                next_segment_id=next_segment_id,
                source="mask2former_small_parent_copied",
            )
            continue

        if parent_label not in REFINABLE_LABELS:
            next_segment_id = copy_parent_segment(
                new_segmap=new_segmap,
                new_segments=new_segments,
                parent_segment=parent_segment,
                parent_mask=parent_mask,
                parent_id=parent_id,
                parent_label=parent_label,
                next_segment_id=next_segment_id,
                source="mask2former_copied",
            )
            continue

        bbox = bbox_from_mask(parent_mask)
        if bbox is None:
            continue

        x1, y1, x2, y2 = pad_bbox(
            bbox=bbox,
            width=width,
            height=height,
            padding=CROP_PADDING,
        )

        crop = image.crop((x1, y1, x2, y2))
        parent_crop_mask = parent_mask[y1:y2, x1:x2]

        candidates = collect_sam_candidates(
            crop=crop,
            parent_crop_mask=parent_crop_mask,
            full_shape=(height, width),
            crop_offset=(x1, y1),
            parent_label=parent_label,
            model=model,
            processor=processor,
        )

        accepted_children = accept_non_overlapping_candidates(
            candidates=candidates,
            parent_mask=parent_mask,
        )

        if not accepted_children:
            if KEEP_ORIGINAL_IF_NO_CHILDREN:
                next_segment_id = copy_parent_segment(
                    new_segmap=new_segmap,
                    new_segments=new_segments,
                    parent_segment=parent_segment,
                    parent_mask=parent_mask,
                    parent_id=parent_id,
                    parent_label=parent_label,
                    next_segment_id=next_segment_id,
                    source="mask2former_fallback_no_sam3",
                )
            continue

        created_child_ids: List[int] = []

        for child in accepted_children:
            child_mask = child["mask"]
            current_child_id = next_segment_id
            new_segmap[child_mask] = current_child_id

            record = segment_record_from_mask(
                segment_id=current_child_id,
                label=child["label"],
                mask=child_mask,
                score=child["score"],
                parent_segment_id=parent_id,
                parent_label=parent_label,
                sam3_prompt=child["prompt"],
                source=child["source"],
            )

            if record:
                new_segments.append(record)
                created_child_ids.append(current_child_id)
                next_segment_id += 1

        # The old parent remainder is assigned to the nearest child.
        # This avoids keeping coarse ring segments around the new SAM3 children.
        new_segmap = absorb_remainder_into_children(
            new_segmap=new_segmap,
            parent_mask=parent_mask,
            child_ids=created_child_ids,
        )

    # Safety step: no valid old pixel should become void in the new segmentation.
    new_segmap, new_segments, next_segment_id, safety_filled_pixels = fill_remaining_old_pixels(
        old_segmap=old_segmap,
        new_segmap=new_segmap,
        new_segments=new_segments,
        old_segments_by_id=old_segments_by_id,
        next_segment_id=next_segment_id,
    )

    if safety_filled_pixels > 0:
        print(f"[INFO] {json_path.name}: safety-filled {safety_filled_pixels} pixels.")

    # Final cleanup and JSON refresh.
    new_segmap = final_clean_sam3_segmap(new_segmap)
    new_segmap = remove_suspicious_label_segments(new_segmap, new_segments)
    new_segments = refresh_segment_records_from_segmap(new_segmap, new_segments)

    debug_check_segmap(
        json_path=json_path,
        old_segmap=old_segmap,
        new_segmap=new_segmap,
        new_segments=new_segments,
    )

    segmap_file = f"{image_id}_sam3_corrected_segmap.npz"
    out_segmap_path = OUT_SEGMAP_DIR / segmap_file
    save_segmap(out_segmap_path, new_segmap)

    out_data = dict(data)
    out_data["segmentation_map_npy"] = segmap_file
    out_data["segmapPath"] = str(out_segmap_path)
    out_data["segments"] = new_segments

    # Keep this only if older follow-up scripts still read panoptic_segments.
    out_data["panoptic_segments"] = new_segments

    out_data["sam3CorrectionConfig"] = {
        "source_json": str(json_path),
        "source_segmap": str(segmap_path),
        "model": "facebook/sam3",
        "device": DEVICE,
        "purpose": "same-label splitting for selected Mask2Former labels",
        "refinable_labels": sorted(REFINABLE_LABELS),
        "crop_padding": CROP_PADDING,
        "sam3_threshold": SAM3_THRESHOLD,
        "sam3_mask_threshold": SAM3_MASK_THRESHOLD,
        "min_parent_area": MIN_PARENT_AREA,
        "min_child_area": MIN_CHILD_AREA,
        "min_parent_overlap_ratio": MIN_PARENT_OVERLAP_RATIO,
        "clip_children_to_parent": CLIP_CHILDREN_TO_PARENT,
        "keep_original_if_no_children": KEEP_ORIGINAL_IF_NO_CHILDREN,
        "final_remove_tiny_components": FINAL_REMOVE_TINY_COMPONENTS,
        "final_small_component_area": FINAL_SMALL_COMPONENT_AREA,
        "final_small_component_ratio": FINAL_SMALL_COMPONENT_RATIO,
        "final_clean_iterations": FINAL_CLEAN_ITERATIONS,
        "final_fill_zero_voids": FINAL_FILL_ZERO_VOIDS,
        "filter_suspicious_label_segments": FILTER_SUSPICIOUS_LABEL_SEGMENTS,
        "suspicious_label_filters": SUSPICIOUS_LABEL_FILTERS,
        "description": (
            "Mask2Former provides the coarse parent segmentation. "
            "SAM3 is applied only to selected labels such as building and house. "
            "The accepted SAM3 children keep the parent label. "
            "Remaining parent pixels are assigned to the nearest accepted child."
        ),
    }

    out_json_path = OUT_JSON_DIR / json_path.name
    save_json(out_json_path, out_data)


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main() -> None:
    ensure_dir(OUT_JSON_DIR)
    ensure_dir(OUT_SEGMAP_DIR)

    print("SAM3 same-label correction")
    print(f"Input JSON dir:   {SEGMENT_JSON_DIR}")
    print(f"Input segmap dir: {SEGMAP_DIR}")
    print(f"Output JSON dir:  {OUT_JSON_DIR}")
    print(f"Output segmaps:   {OUT_SEGMAP_DIR}")
    print(f"Device:           {DEVICE}")
    print(f"Resume:           {RESUME}")
    print(f"Max files:        {MAX_FILES if MAX_FILES is not None else 'all'}")

    model, processor = load_sam3()

    start_time = time.perf_counter()
    json_files = sorted(SEGMENT_JSON_DIR.glob("*.json"))

    if MAX_FILES is not None:
        json_files = json_files[:MAX_FILES]

    print(f"Found {len(json_files)} Mask2Former JSON files.")

    processed = 0
    skipped = 0
    errors = 0
    error_items: List[Dict[str, str]] = []

    for json_path in tqdm(json_files, desc="SAM3 correcting"):
        out_path = OUT_JSON_DIR / json_path.name

        if RESUME and out_path.exists():
            skipped += 1
            continue

        try:
            process_one_json(
                json_path=json_path,
                model=model,
                processor=processor,
            )
            processed += 1

        except Exception as e:
            errors += 1
            error_items.append({"file": str(json_path), "error": str(e)})
            print(f"[ERROR] {json_path.name}: {e}")

    if error_items:
        error_log_path = OUT_JSON_DIR.parent / ERROR_LOG_NAME
        save_json(error_log_path, {"errors": error_items})
        print(f"Wrote error log: {error_log_path}")

    elapsed = time.perf_counter() - start_time

    print("Done.")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {errors}")
    print(f"Runtime:   {elapsed:.2f} seconds / {elapsed / 60:.2f} minutes / {elapsed / 3600:.2f} hours")


if __name__ == "__main__":
    main()
