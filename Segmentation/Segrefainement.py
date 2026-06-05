import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import open_clip


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

SEGMENT_JSON_DIR = Path(r"Segmentation\output_sam3_correction\json")
SEGMAP_DIR = Path(r"Segmentation\output_sam3_correction\segmaps")

OUT_JSON_DIR = Path(r"Segmentation\output_refined\json")

LABEL_CANDIDATES_FILE = Path("Segmentation\label_candidates.json")

# OpenCLIP model.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPENCLIP_MODEL_NAME = "ViT-B-16"
OPENCLIP_PRETRAINED = "laion2b_s34b_b88k"

# Crop settings.
CROP_PADDING = 20
CONTEXT_PADDING = 80

# Segments smaller than this are not refined.
MIN_SEGMENT_AREA = 100

# Number of keywords saved per segment.
TOP_K_KEYWORDS = 4

# Refinement decision:
REFINEMENT_MARGIN = 0.002
MIN_REFINED_SCORE = 0.18

# Weighting between context crop and isolated segment crop.
CONTEXT_WEIGHT = 0.55
SEGMENT_WEIGHT = 0.45

# Batch sizes for CLIP encoding.
IMAGE_BATCH_SIZE = 96
TEXT_BATCH_SIZE = 256

# Runtime options.
RESUME = True
MAX_FILES: Optional[int] = None

# Keep debug scores in the output.
SAVE_DEBUG_FIELDS = False

# Keep source/parent fields from SAM3 correction in each segment.
KEEP_SOURCE_FIELDS = False

# -------------------------------------------------
# GLOBAL CACHE
# -------------------------------------------------

TEXT_EMBEDDING_CACHE: Dict[Tuple[str, ...], torch.Tensor] = {}

# -------------------------------------------------
# BASIC HELPERS
# -------------------------------------------------

def normalize_text(value: Any) -> str:
    """Normalize labels and candidate names."""
    if value is None:
        return ""
    return str(value).strip().lower()


def unique_keep_order(items: List[Any]) -> List[str]:
    """Remove duplicates while keeping the original order."""
    seen = set()
    result: List[str] = []

    for item in items:
        item_norm = normalize_text(item)
        if not item_norm or item_norm in seen:
            continue
        seen.add(item_norm)
        result.append(item_norm)

    return result


def ensure_dir(path: Path):
    """Create a folder if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]):
    """Save a JSON file."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------

def load_openclip():
    """Load OpenCLIP model, preprocessing and tokenizer."""
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, _, preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL_NAME,
        pretrained=OPENCLIP_PRETRAINED,
        device=DEVICE,
    )
    model.eval()

    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL_NAME)
    return model, preprocess, tokenizer


# -------------------------------------------------
# INPUT LOADING
# -------------------------------------------------

def load_label_candidates(path: Path) -> Dict[str, List[str]]:
    """Load possible refined labels for every original segment label."""
    if not path.exists():
        raise FileNotFoundError(
            f"Label candidates file not found: {path}. "
            f"Create it first, for example with create_label_candidates.py."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized: Dict[str, List[str]] = {}

    for label, candidates in data.items():
        label_norm = normalize_text(label)

        if not isinstance(candidates, list):
            candidates = [label_norm]

        clean_candidates = unique_keep_order(candidates)

        # Always include the original label as a possible candidate.
        clean_candidates.append(label_norm)
        normalized[label_norm] = unique_keep_order(clean_candidates)

    return normalized


def load_segmap(path: Path) -> np.ndarray:
    """Load a segmentation map from npz or npy."""
    obj = np.load(path, allow_pickle=False)

    if isinstance(obj, np.lib.npyio.NpzFile):
        try:
            # Prefer common keys, otherwise use the first array.
            for key in ["segmap", "arr_0", "segmentation", "mask", "masks"]:
                if key in obj.files:
                    arr = obj[key]
                    break
            else:
                arr = obj[obj.files[0]]
        finally:
            obj.close()

        return np.asarray(arr)

    return np.asarray(obj)


def resolve_image_path(data: Dict[str, Any]) -> Path:
    """Read the image path from the fixed pipeline JSON format."""
    image_path = data.get("image_path") or data.get("imagePath")
    if not image_path:
        raise ValueError("No image_path found in JSON.")

    return Path(str(image_path))


def resolve_segmap_path(data: Dict[str, Any]) -> Path:
    """
    Resolve the segmap path.

    The JSON may contain only the file name, for example:
    10023_img060_sam3_corrected_segmap.npz

    In that case the file is searched inside SEGMAP_DIR.
    """
    value = (
        data.get("segmentation_map_npy")
        or data.get("segmapPath")
        or data.get("segmap_path")
    )

    if not value:
        raise ValueError("No segmentation_map_npy or segmapPath found in JSON.")

    p = Path(str(value))

    if p.exists():
        return p

    p_in_segmap_dir = SEGMAP_DIR / p.name
    if p_in_segmap_dir.exists():
        return p_in_segmap_dir

    raise FileNotFoundError(f"Segmap not found: {p} or {p_in_segmap_dir}")


def get_segments(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read the segment list from the fixed pipeline JSON format."""
    segments = data.get("segments")

    if not isinstance(segments, list):
        raise ValueError("No valid 'segments' list found in JSON.")

    return segments


def read_bbox(segment: Dict[str, Any], image_width: int, image_height: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Read bbox from the segment.

    Your JSON uses:
    bbox = {x1, y1, x2, y2, width, height}

    x2 and y2 are treated as inclusive coordinates, therefore +1 is used
    for NumPy slicing and PIL crop logic.
    """
    bbox = segment.get("bbox")
    if not isinstance(bbox, dict):
        return None

    try:
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"]) + 1
        y2 = int(bbox["y2"]) + 1
    except Exception:
        return None

    x1 = max(0, min(image_width, x1))
    y1 = max(0, min(image_height, y1))
    x2 = max(0, min(image_width, x2))
    y2 = max(0, min(image_height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


# -------------------------------------------------
# IMAGE / MASK HELPERS
# -------------------------------------------------

def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Calculate a bbox directly from a binary mask."""
    ys, xs = np.nonzero(mask)

    if xs.size == 0 or ys.size == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    padding: int,
) -> Tuple[int, int, int, int]:
    """Add padding around a bbox and keep it inside the image."""
    x1, y1, x2, y2 = bbox

    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def make_crops_from_segmap(
    image: Image.Image,
    image_np: np.ndarray,
    segmap: np.ndarray,
    segment_id: int,
    bbox_hint: Optional[Tuple[int, int, int, int]],
) -> Tuple[Optional[Image.Image], Optional[Image.Image], int]:
    """
    Create two crops for one segment:
    1. segment_crop: only the segment, outside pixels are black
    2. context_crop: original image crop around the segment
    """
    width, height = image.size

    # Use bbox from JSON if possible to avoid scanning the full segmap.
    if bbox_hint is not None:
        x1, y1, x2, y2 = bbox_hint
        local_mask = segmap[y1:y2, x1:x2] == segment_id

        if not local_mask.any():
            return None, None, 0

        area = int(local_mask.sum())
        bbox = bbox_hint
    else:
        full_mask = segmap == segment_id
        area = int(full_mask.sum())
        bbox = bbox_from_mask(full_mask)

        if bbox is None:
            return None, None, 0

    # Segment crop with black background.
    sx1, sy1, sx2, sy2 = pad_bbox(bbox, width, height, CROP_PADDING)
    crop_np = image_np[sy1:sy2, sx1:sx2].copy()
    crop_mask = segmap[sy1:sy2, sx1:sx2] == segment_id
    crop_np[~crop_mask] = 0
    segment_crop = Image.fromarray(crop_np)

    # Context crop from original image.
    cx1, cy1, cx2, cy2 = pad_bbox(bbox, width, height, CONTEXT_PADDING)
    context_crop = image.crop((cx1, cy1, cx2, cy2))

    return segment_crop, context_crop, area


# -------------------------------------------------
# CLIP ENCODING
# -------------------------------------------------

@torch.inference_mode()
def encode_images_clip_batched(
    images: List[Image.Image],
    model,
    preprocess,
    batch_size: int = IMAGE_BATCH_SIZE,
) -> torch.Tensor:
    """Encode image crops with OpenCLIP in batches."""
    if not images:
        return torch.empty((0, 0), device=DEVICE)

    all_embs = []

    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]

        image_tensor = torch.stack(
            [preprocess(img.convert("RGB")) for img in batch]
        ).to(DEVICE, non_blocking=True)

        emb = model.encode_image(image_tensor)

        # Normalize embeddings so dot product equals cosine similarity.
        emb = F.normalize(emb.float(), dim=-1)
        all_embs.append(emb)

    return torch.cat(all_embs, dim=0)


@torch.inference_mode()
def encode_texts_clip(
    texts: List[str],
    model,
    tokenizer,
) -> torch.Tensor:
    """Encode text prompts with OpenCLIP in batches."""
    if not texts:
        return torch.empty((0, 0), device=DEVICE)

    all_embs = []

    for start in range(0, len(texts), TEXT_BATCH_SIZE):
        batch = texts[start:start + TEXT_BATCH_SIZE]
        tokens = tokenizer(batch).to(DEVICE, non_blocking=True)

        emb = model.encode_text(tokens)

        # Normalize embeddings so dot product equals cosine similarity.
        emb = F.normalize(emb.float(), dim=-1)
        all_embs.append(emb)

    return torch.cat(all_embs, dim=0)


@torch.inference_mode()
def get_text_embeddings_cached(
    candidates: List[str],
    model,
    tokenizer,
) -> torch.Tensor:
    """Return cached text embeddings for a candidate list."""
    candidates = unique_keep_order(candidates)
    cache_key = tuple(candidates)

    cached = TEXT_EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached

    prompts = [f"a photo of {candidate}" for candidate in candidates]
    text_emb = encode_texts_clip(prompts, model=model, tokenizer=tokenizer)

    TEXT_EMBEDDING_CACHE[cache_key] = text_emb
    return text_emb


def ranked_from_scores(
    candidates: List[str],
    score_row: torch.Tensor,
) -> List[Tuple[str, float]]:
    """Sort candidates by CLIP score descending."""
    k = len(candidates)
    values, indices = torch.topk(score_row, k=k)

    return [
        (candidates[idx], float(score))
        for score, idx in zip(values.tolist(), indices.tolist())
    ]


# -------------------------------------------------
# REFINEMENT LOGIC
# -------------------------------------------------

def get_label_candidates(
    label: str,
    label_candidates: Dict[str, List[str]],
) -> List[str]:
    """Get all possible refined labels for one original label."""
    label_norm = normalize_text(label)
    candidates = label_candidates.get(label_norm, [label_norm]).copy()

    # Always include the original label.
    candidates.append(label_norm)

    return unique_keep_order(candidates)


def combine_rankings(
    ranked_context: List[Tuple[str, float]],
    ranked_segment: List[Tuple[str, float]],
) -> List[Tuple[str, float]]:
    """Combine context and segment scores into one ranking."""
    combined_scores: Dict[str, float] = {}

    for candidate, score in ranked_context:
        candidate = normalize_text(candidate)
        combined_scores[candidate] = combined_scores.get(candidate, 0.0) + CONTEXT_WEIGHT * float(score)

    for candidate, score in ranked_segment:
        candidate = normalize_text(candidate)
        combined_scores[candidate] = combined_scores.get(candidate, 0.0) + SEGMENT_WEIGHT * float(score)

    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)


def choose_refined_label(
    original_label: str,
    ranked: List[Tuple[str, float]],
) -> Tuple[Optional[str], Optional[float], str, str]:
    """
    Decide whether a segment gets a refined label.

    A refined label is accepted only if:
    - it is different from the original label
    - its score is at least MIN_REFINED_SCORE
    - it beats the original label by REFINEMENT_MARGIN
    """
    original_label = normalize_text(original_label)

    if not ranked:
        return None, None, "no candidates", "none"

    original_score = 0.0

    for candidate, score in ranked:
        if normalize_text(candidate) == original_label:
            original_score = float(score)
            break

    best_different_label = None
    best_different_score = None

    for candidate, score in ranked:
        candidate = normalize_text(candidate)
        score = float(score)

        if candidate == original_label:
            continue

        if score < MIN_REFINED_SCORE:
            continue

        best_different_label = candidate
        best_different_score = score
        break

    if best_different_label is None:
        return None, None, "no different candidate found", "none"

    score_margin = float(best_different_score) - float(original_score)

    if score_margin >= REFINEMENT_MARGIN:
        return (
            best_different_label,
            best_different_score,
            f"refined because margin was reached ({score_margin:.4f} >= {REFINEMENT_MARGIN:.4f})",
            "high",
        )

    return (
        None,
        None,
        f"not refined because margin was too small ({score_margin:.4f} < {REFINEMENT_MARGIN:.4f})",
        "none",
    )


def build_keywords_from_ranked(
    original_label: str,
    refined_label: Optional[str],
    ranked: List[Tuple[str, float]],
) -> Tuple[List[str], List[float]]:
    """
    Build the keyword list for one segment.

    The refined label is added first if it exists.
    The original label is always included.
    """
    original_label = normalize_text(original_label)
    refined_label_norm = normalize_text(refined_label) if refined_label else None

    keywords: List[str] = []
    scores: List[float] = []

    def add_keyword(candidate: str, score: float):
        candidate = normalize_text(candidate)

        if candidate and candidate not in keywords and len(keywords) < TOP_K_KEYWORDS:
            keywords.append(candidate)
            scores.append(float(score))

    # Put refined label first.
    if refined_label_norm:
        refined_score = next(
            (
                float(score)
                for candidate, score in ranked
                if normalize_text(candidate) == refined_label_norm
            ),
            1.0,
        )
        add_keyword(refined_label_norm, refined_score)

    # Add best other candidates.
    for candidate, score in ranked:
        candidate = normalize_text(candidate)

        if candidate != original_label:
            add_keyword(candidate, float(score))

        # Leave space for original label.
        if len(keywords) >= TOP_K_KEYWORDS - 1:
            break

    # Always include original label.
    if original_label not in keywords:
        original_score = next(
            (
                float(score)
                for candidate, score in ranked
                if normalize_text(candidate) == original_label
            ),
            1.0,
        )
        add_keyword(original_label, original_score)

    return keywords[:TOP_K_KEYWORDS], scores[:TOP_K_KEYWORDS]


# -------------------------------------------------
# OUTPUT HELPERS
# -------------------------------------------------

def build_clean_segment(
    segment: Dict[str, Any],
    original_label: str,
    refined_label: Optional[str],
    refined_score: Optional[float],
    display_label: str,
    refinement_confidence: str,
    refinement_reason: str,
    keywords: List[str],
    keyword_scores: List[float],
    real_area: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a clean segment object for the output JSON.

    This avoids writing unnecessary fields like:
    - source
    - parent_segment_id
    - parent_label
    - sam3_prompt
    unless KEEP_SOURCE_FIELDS is enabled.
    """
    clean: Dict[str, Any] = {
        "segment_id": segment.get("segment_id"),
        "category_id": segment.get("category_id"),
        "label": original_label,
        "originalLabel": original_label,
        "refinedLabel": refined_label,
        "refinedLabelScore": refined_score,
        "displayLabel": display_label,
        "refinementConfidence": refinement_confidence,
        "refinementReason": refinement_reason,
        "keywords": keywords,
        "keywordScores": keyword_scores,
        "area": int(real_area if real_area is not None else segment.get("area", 0)),
        "score": segment.get("score"),
        "was_fused": segment.get("was_fused", False),
        "bbox": segment.get("bbox"),
    }

    if KEEP_SOURCE_FIELDS:
        for key in ["source", "parent_segment_id", "parent_label", "sam3_prompt"]:
            if key in segment:
                clean[key] = segment[key]

    return clean


def build_unrefined_segment(
    segment: Dict[str, Any],
    label: str,
    reason: str,
    real_area: Optional[int] = None,
) -> Dict[str, Any]:
    """Build output for a segment that was skipped or not refined."""
    return build_clean_segment(
        segment=segment,
        original_label=label,
        refined_label=None,
        refined_score=None,
        display_label=label,
        refinement_confidence="none",
        refinement_reason=reason,
        keywords=[label],
        keyword_scores=[1.0],
        real_area=real_area,
    )


def build_output_json(
    input_data: Dict[str, Any],
    refined_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the final clean JSON.

    This intentionally does not copy:
    - panoptic_segments
    - postprocess
    - sam3CorrectionConfig

    These fields describe how the file was produced, but they are not needed
    for the refined output used by similarity calculation, Neo4j or frontend.
    """
    return {
        "image_id": input_data.get("image_id"),
        "image_path": input_data.get("image_path") or input_data.get("imagePath"),
        "width": input_data.get("width"),
        "height": input_data.get("height"),
        "segmentation_map_npy": input_data.get("segmentation_map_npy"),
        "segmapPath": input_data.get("segmapPath"),
        "segments": refined_segments,
        "refinementConfig": {
            "openclip_model": OPENCLIP_MODEL_NAME,
            "openclip_pretrained": OPENCLIP_PRETRAINED,
            "label_candidates_file": str(LABEL_CANDIDATES_FILE),
            "top_k_keywords": TOP_K_KEYWORDS,
            "crop_padding": CROP_PADDING,
            "context_padding": CONTEXT_PADDING,
            "min_segment_area": MIN_SEGMENT_AREA,
            "refinement_margin": REFINEMENT_MARGIN,
            "min_refined_score": MIN_REFINED_SCORE,
            "context_weight": CONTEXT_WEIGHT,
            "segment_weight": SEGMENT_WEIGHT,
        },
    }


# -------------------------------------------------
# MAIN PROCESSING
# -------------------------------------------------

def refine_one_json(
    json_path: Path,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    label_candidates: Dict[str, List[str]],
):
    """Refine all segments of one JSON file."""
    data = load_json(json_path)

    image_path = resolve_image_path(data)
    segmap_path = resolve_segmap_path(data)

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    segmap = load_segmap(segmap_path)

    image_width, image_height = image.size

    if segmap.ndim > 2:
        segmap = np.squeeze(segmap)

    if segmap.shape[0] != image_height or segmap.shape[1] != image_width:
        raise ValueError(
            f"Segmap size does not match image. "
            f"segmap={segmap.shape}, image={(image_width, image_height)}"
        )

    segments = get_segments(data)

    refined_segments: List[Dict[str, Any]] = []
    pending_jobs: List[Dict[str, Any]] = []
    clip_images: List[Image.Image] = []

    # First pass:
    # Create crops for every usable segment.
    for segment in segments:
        try:
            segment_id = int(segment["segment_id"])
            label = normalize_text(segment["label"])
        except Exception:
            continue

        bbox_hint = read_bbox(
            segment=segment,
            image_width=image_width,
            image_height=image_height,
        )

        segment_crop, context_crop, real_area = make_crops_from_segmap(
            image=image,
            image_np=image_np,
            segmap=segmap,
            segment_id=segment_id,
            bbox_hint=bbox_hint,
        )

        if real_area <= 0:
            # Fallback to JSON area if the mask cannot be read correctly.
            real_area = int(segment.get("area", 0))

        if real_area < MIN_SEGMENT_AREA:
            refined_segments.append(
                build_unrefined_segment(
                    segment=segment,
                    label=label,
                    reason="segment too small",
                    real_area=real_area,
                )
            )
            continue

        if segment_crop is None or context_crop is None:
            refined_segments.append(
                build_unrefined_segment(
                    segment=segment,
                    label=label,
                    reason="no crop",
                    real_area=real_area,
                )
            )
            continue

        candidates = get_label_candidates(label, label_candidates)

        # Every segment gets two images:
        # 1. context crop
        # 2. isolated segment crop
        image_index_context = len(clip_images)
        clip_images.append(context_crop)

        image_index_segment = len(clip_images)
        clip_images.append(segment_crop)

        pending_jobs.append(
            {
                "segment": segment,
                "label": label,
                "real_area": real_area,
                "candidates": candidates,
                "candidate_key": tuple(candidates),
                "image_index_context": image_index_context,
                "image_index_segment": image_index_segment,
            }
        )

    # Encode all crops in one batched run.
    image_embs = encode_images_clip_batched(
        images=clip_images,
        model=clip_model,
        preprocess=clip_preprocess,
        batch_size=IMAGE_BATCH_SIZE,
    ) if clip_images else None

    # Group segments with the same candidate list.
    # This avoids repeated text encoding and many tiny matrix multiplications.
    jobs_by_candidates: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}

    for job in pending_jobs:
        jobs_by_candidates.setdefault(job["candidate_key"], []).append(job)

    # Second pass:
    # Compare crop embeddings with candidate label embeddings.
    for candidate_key, jobs in jobs_by_candidates.items():
        candidates = list(candidate_key)

        text_emb = get_text_embeddings_cached(
            candidates,
            model=clip_model,
            tokenizer=clip_tokenizer,
        )

        context_indices = [job["image_index_context"] for job in jobs]
        segment_indices = [job["image_index_segment"] for job in jobs]

        context_scores_all = image_embs[context_indices] @ text_emb.T
        segment_scores_all = image_embs[segment_indices] @ text_emb.T

        for i, job in enumerate(jobs):
            segment = job["segment"]
            label = job["label"]
            real_area = job["real_area"]

            ranked_context = ranked_from_scores(candidates, context_scores_all[i])
            ranked_segment = ranked_from_scores(candidates, segment_scores_all[i])
            ranked = combine_rankings(ranked_context, ranked_segment)

            refined_label, refined_score, refinement_reason, refinement_confidence = choose_refined_label(
                original_label=label,
                ranked=ranked,
            )

            display_label = refined_label if refined_label is not None else label

            keywords, keyword_scores = build_keywords_from_ranked(
                original_label=label,
                refined_label=refined_label,
                ranked=ranked,
            )

            clean_segment = build_clean_segment(
                segment=segment,
                original_label=label,
                refined_label=refined_label,
                refined_score=refined_score,
                display_label=display_label,
                refinement_confidence=refinement_confidence,
                refinement_reason=refinement_reason,
                keywords=keywords,
                keyword_scores=keyword_scores,
                real_area=real_area,
            )

            if SAVE_DEBUG_FIELDS:
                clean_segment["allowedCandidates"] = candidates
                clean_segment["clipContextCandidates"] = [candidate for candidate, _ in ranked_context]
                clean_segment["clipContextScores"] = [score for _, score in ranked_context]
                clean_segment["clipSegmentCandidates"] = [candidate for candidate, _ in ranked_segment]
                clean_segment["clipSegmentScores"] = [score for _, score in ranked_segment]
                clean_segment["clipCombinedCandidates"] = [candidate for candidate, _ in ranked]
                clean_segment["clipCombinedScores"] = [score for _, score in ranked]

            refined_segments.append(clean_segment)

    output_data = build_output_json(
        input_data=data,
        refined_segments=refined_segments,
    )

    save_json(OUT_JSON_DIR / json_path.name, output_data)


def main():
    """Run refinement for all JSON files."""
    ensure_dir(OUT_JSON_DIR)
    start_time = time.perf_counter()

    print(f"Using device: {DEVICE}")

    if DEVICE == "cuda":
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

    print("Loading label candidates...")
    label_candidates = load_label_candidates(LABEL_CANDIDATES_FILE)
    print(f"Loaded candidates for {len(label_candidates)} labels.")

    print("Loading OpenCLIP...")
    clip_model, clip_preprocess, clip_tokenizer = load_openclip()

    json_files = sorted(SEGMENT_JSON_DIR.glob("*.json"))

    if MAX_FILES is not None:
        json_files = json_files[:MAX_FILES]

    print(f"Found {len(json_files)} JSON files.")

    processed = 0
    skipped = 0
    errors = 0

    for json_path in tqdm(json_files, desc="Refining segments"):
        out_path = OUT_JSON_DIR / json_path.name

        if RESUME and out_path.exists():
            skipped += 1
            continue

        try:
            refine_one_json(
                json_path=json_path,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                clip_tokenizer=clip_tokenizer,
                label_candidates=label_candidates,
            )
            processed += 1

        except Exception as e:
            errors += 1
            print(f"[ERROR] {json_path.name}: {e}")

    elapsed = time.perf_counter() - start_time

    print("Done.")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {errors}")
    print(f"Runtime:   {elapsed:.2f} seconds")
    print(f"Runtime:   {elapsed / 60:.2f} minutes")
    print(f"Runtime:   {elapsed / 3600:.2f} hours")
    print(f"Text embedding cache entries: {len(TEXT_EMBEDDING_CACHE)}")


if __name__ == "__main__":
    main()
