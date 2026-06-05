import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip

# ============================================================
# Konfiguration
# ============================================================

# Eingabe- und Ausgabeordner. Diese Werte müssen normalerweise angepasst werden.
SEGMENTS_JSON_DIR = r"Segmentation\output_refined\json"
SEGMAPS_DIR = r"Segmentation\output_sam3_correction\segmaps"
SEGMENT_TAXONOMY_JSON = r"Segmentation\KeywordsSeg.json"
OUT_DIR = r"Segmentation\output_segmentation_emb"

# Nur nötig, wenn image_path in den JSON-Dateien relativ gespeichert ist.
IMAGES_BASE_DIR = None

# CLIP-Modell und Laufzeiteinstellungen.
MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"
DEVICE = "auto"          # "auto", "cuda", or "cpu"
IMAGE_BATCH_SIZE = 64    # bei zu wenig CUDA-Speicher auf 32 oder 16 reduzieren
TEXT_BATCH_SIZE = 256
USE_AMP = True           # schneller auf CUDA, normalerweise ohne sichtbaren Qualitätsverlust

# Einstellungen für die Konzeptextraktion.
TOP_KW_SEGMENT = 5
CONCEPT_THRESHOLD = 0.18
MAX_CONCEPTS_PER_SEGMENT = 8

# Grenzen für Kandidaten und Kanten.
MAX_SEGMENTS_PER_CONCEPT = 500
MAX_CANDIDATES_PER_SEGMENT = 700
EDGES_PER_SEGMENT = 10

# Gewichtung für den finalen Ähnlichkeitswert.
W_SEGMENT = 0.50
W_CONCEPT = 0.30
W_REFINED_LABEL = 0.20

# Filtereinstellungen für Vergleichskandidaten.
USE_CONDITIONED_PROMPTS = False
USE_CONCEPT_PREFILTER = True
EXCLUDE_SAME_IMAGE = True
COMPARE_ONLY_SAME_LABEL = True

# Einstellungen für Segment-Crops und Grössenfilter.
CROP_PAD = 4
BLACK_BACKGROUND = True
MIN_AREA = 300
MIN_AREA_RATIO = 0.0
USE_LOW_CONFIDENCE_REFINED = False



# ============================================================
# Hilfsfunktionen
# ============================================================

def normalize_label(value: Any) -> str:
    return str(value).strip().lower()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_single_json(path: str | Path) -> Dict[str, Any] | List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_json_records_from_folder(json_dir: str | Path) -> List[Dict[str, Any]]:
    json_dir = Path(json_dir)
    records: List[Dict[str, Any]] = []

    for path in sorted(json_dir.glob("*.json")):
        try:
            data = load_single_json(path)
            if isinstance(data, dict):
                records.append(data)
            elif isinstance(data, list):
                records.extend([x for x in data if isinstance(x, dict)])
            else:
                print(f"[WARN] Skipping {path}: unsupported JSON root")
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")

    return records


def load_segmap_array(segmap_path: str | Path) -> np.ndarray:
    data = np.load(segmap_path)

    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, np.lib.npyio.NpzFile):
        try:
            preferred_keys = ("segmap", "arr_0", "mask", "data")
            for key in preferred_keys:
                if key in data.files:
                    return data[key]
            if data.files:
                return data[data.files[0]]
            raise ValueError(f"Empty npz file: {segmap_path}")
        finally:
            data.close()

    raise TypeError(f"Unsupported segmap type: {type(data)}")


def resolve_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if base_dir:
        return Path(base_dir) / path
    return path


def topk_from_scores(scores: np.ndarray, labels: Sequence[str], k: int) -> Tuple[List[str], List[float]]:
    if k <= 0 or scores.size == 0:
        return [], []

    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [labels[int(i)] for i in idx], [float(scores[int(i)]) for i in idx]


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def get_original_and_refined_labels(
    seg: Dict[str, Any],
    use_low_confidence_refined: bool = True,
) -> Tuple[str, str, str]:
    original_label = normalize_label(seg.get("originalLabel") or seg.get("label") or "")
    refined_label = normalize_label(seg.get("refinedLabel") or "")
    refinement_confidence = normalize_label(seg.get("refinementConfidence") or "none")

    if refined_label:
        if refinement_confidence == "low" and not use_low_confidence_refined:
            semantic_label = original_label
        else:
            semantic_label = refined_label
    else:
        semantic_label = original_label


    return original_label, refined_label, semantic_label


def should_keep_segment(
    seg: Dict[str, Any],
    min_area: int,
    min_area_ratio: float,
    image_w: int,
    image_h: int,
) -> bool:
    label = normalize_label(seg.get("originalLabel") or seg.get("label") or "")
    area = int(seg.get("area", 0))

    if not label:
        return False
    if area < min_area:
        return False

    total_area = max(1, image_w * image_h)
    if area / total_area < min_area_ratio:
        return False

    return True


# ============================================================
# Segment-Crops und CLIP-Embeddings
# ============================================================

def crop_from_mask(
    img: Image.Image,
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int] | None = None,
    pad: int = 0,
    black_background: bool = True,
) -> Image.Image:
    img_np = np.asarray(img.convert("RGB"))
    h, w = mask.shape[:2]
    mask = mask > 0

    if bbox is None:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Empty mask")
        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1
    else:
        x1, y1, x2, y2 = bbox

    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid mask crop bbox: {(x1, y1, x2, y2)}")

    crop = img_np[y1:y2, x1:x2].copy()
    if black_background:
        crop_mask = mask[y1:y2, x1:x2]
        crop[~crop_mask] = 0

    return Image.fromarray(crop)


@torch.no_grad()
def embed_pil_images_in_chunks(
    model: torch.nn.Module,
    preprocess,
    pil_imgs: Sequence[Image.Image],
    device: str,
    batch_size: int,
    use_amp: bool,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []

    for imgs in batched(pil_imgs, batch_size):
        batch = torch.stack([preprocess(img.convert("RGB")) for img in imgs]).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(use_amp and device.startswith("cuda"))):
            feats = model.encode_image(batch)
            feats = F.normalize(feats.float(), dim=-1)
        outputs.append(feats.cpu())

    return torch.cat(outputs, dim=0)


@torch.no_grad()
def embed_text_prompts_in_chunks(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    device: str,
    batch_size: int,
    use_amp: bool,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []

    for prompt_batch in batched(prompts, batch_size):
        tokens = tokenizer(list(prompt_batch)).to(device)
        with torch.autocast(device_type="cuda", enabled=(use_amp and device.startswith("cuda"))):
            feats = model.encode_text(tokens)
            feats = F.normalize(feats.float(), dim=-1)
        outputs.append(feats.cpu())

    return torch.cat(outputs, dim=0)


# ============================================================
# Konzept-Prompts und Text-Embeddings
# ============================================================

def get_templates_for_label(
    label: str,
    color_labels: Set[str],
    material_labels: Set[str],
    attribute_labels: Set[str],
    base_segment_label: str | None = None,
) -> List[str]:
    if label in color_labels:
        if base_segment_label:
            return [
                f"a {label} {base_segment_label}",
                f"a scene segment showing {label} {base_segment_label}",
                f"a region of {base_segment_label} that is {label}",
            ]
        return [
            "a {} region",
            "a scene area that is {}",
            "a segmented region that is {}",
        ]

    if label in material_labels:
        if base_segment_label:
            return [
                f"a {base_segment_label} made of {label}",
                f"a {label} {base_segment_label}",
                f"a segmented region of {base_segment_label} with {label} material",
            ]
        return [
            "a region made of {}",
            "a {} surface",
            "a segmented region with {} material",
        ]

    if label in attribute_labels:
        if base_segment_label:
            return [
                f"a {label} {base_segment_label}",
                f"a {base_segment_label} that looks {label}",
                f"a scene segment of a {base_segment_label} that is {label}",
            ]
        return [
            "a {} region",
            "a region that looks {}",
            "a segmented region that is {}",
        ]

    return [
        "a photo of {}",
        "an image region of {}",
        "a segmented region of {}",
    ]


@torch.no_grad()
def build_text_matrix_fast(
    model: torch.nn.Module,
    tokenizer,
    keyword_taxonomy: Sequence[str],
    device: str,
    color_labels: Set[str],
    material_labels: Set[str],
    attribute_labels: Set[str],
    batch_size: int,
    use_amp: bool,
    base_segment_label: str | None = None,
) -> torch.Tensor:
    """
    Faster than encoding one keyword after the other.
    Encodes all prompts for one matrix in text batches, then averages each keyword's templates.
    """
    all_prompts: List[str] = []
    groups: List[Tuple[int, int]] = []

    for label in keyword_taxonomy:
        templates = get_templates_for_label(
            label=label,
            color_labels=color_labels,
            material_labels=material_labels,
            attribute_labels=attribute_labels,
            base_segment_label=base_segment_label,
        )
        start = len(all_prompts)
        all_prompts.extend([t.format(label) if "{}" in t else t for t in templates])
        groups.append((start, len(all_prompts)))

    prompt_feats = embed_text_prompts_in_chunks(
        model=model,
        tokenizer=tokenizer,
        prompts=all_prompts,
        device=device,
        batch_size=batch_size,
        use_amp=use_amp,
    )

    matrix: List[torch.Tensor] = []
    for start, end in groups:
        feat = prompt_feats[start:end].mean(dim=0)
        feat = F.normalize(feat, dim=-1)
        matrix.append(feat)

    return torch.stack(matrix, dim=0).to(device)


@torch.no_grad()
def embed_unique_labels_fast(
    model: torch.nn.Module,
    tokenizer,
    labels: Sequence[str],
    device: str,
    batch_size: int,
    use_amp: bool,
) -> Dict[str, torch.Tensor]:
    unique_labels = sorted({label for label in labels if label})
    if not unique_labels:
        return {}

    prompts = [f"a photo of {label}" for label in unique_labels]
    feats = embed_text_prompts_in_chunks(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        batch_size=batch_size,
        use_amp=use_amp,
    )

    return {label: feats[i] for i, label in enumerate(unique_labels)}


def select_best_concepts_by_category(
    scores: np.ndarray,
    keyword_taxonomy: Sequence[str],
    color_labels: Set[str],
    material_labels: Set[str],
    attribute_labels: Set[str],
    concept_threshold: float,
    max_colors: int = 1,
    max_materials: int = 1,
    max_attributes: int = 2,
) -> List[int]:
    color_candidates: List[int] = []
    material_candidates: List[int] = []
    attribute_candidates: List[int] = []

    for i, label in enumerate(keyword_taxonomy):
        score = float(scores[i])
        if score < concept_threshold:
            continue

        if label in color_labels:
            color_candidates.append(i)
        elif label in material_labels:
            material_candidates.append(i)
        elif label in attribute_labels:
            attribute_candidates.append(i)

    color_candidates = sorted(color_candidates, key=lambda i: scores[i], reverse=True)[:max_colors]
    material_candidates = sorted(material_candidates, key=lambda i: scores[i], reverse=True)[:max_materials]
    attribute_candidates = sorted(attribute_candidates, key=lambda i: scores[i], reverse=True)[:max_attributes]

    selected = color_candidates + material_candidates + attribute_candidates
    return sorted(selected, key=lambda i: scores[i], reverse=True)


# ============================================================
# Pipeline für Segmentähnlichkeit
# ============================================================

def main(
    segments_json_dir: str,
    segmaps_dir: str,
    segment_taxonomy_json_path: str,
    out_dir: str,
    images_base_dir: Optional[str] = None,
    model_name: str = "ViT-B-16",
    pretrained: str = "laion2b_s34b_b88k",
    device: str = "auto",
    image_batch_size: int = 64,
    text_batch_size: int = 256,
    use_amp: bool = True,
    top_kw_segment: int = 12,
    concept_threshold: float = 0.20,
    max_concepts_per_segment: int = 8,
    max_segments_per_concept: int = 1500,
    max_candidates_per_segment: int = 1000,
    edges_per_segment: int = 50,
    w_segment: float = 0.60,
    w_concept: float = 0.30,
    w_refined_label: float = 0.10,
    use_conditioned_prompts: bool = True,
    use_concept_prefilter: bool = True,
    exclude_same_image: bool = True,
    compare_only_same_label: bool = True,
    crop_pad: int = 4,
    black_background: bool = True,
    min_area: int = 300,
    min_area_ratio: float = 0.0,
    use_low_confidence_refined: bool = True,
) -> None:
    start_time = time.perf_counter()
    out_dir_p = Path(out_dir)
    ensure_dir(out_dir_p)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")
    print(f"Image batch size: {image_batch_size}")
    print(f"Text batch size: {text_batch_size}")
    print(f"AMP enabled: {use_amp and device.startswith('cuda')}")

    with open(segment_taxonomy_json_path, "r", encoding="utf-8-sig") as f:
        taxonomy_data = json.load(f)

    required_keys = ("COLOR_LABELS", "MATERIAL_LABELS", "ATTRIBUTE_LABELS")
    for key in required_keys:
        if key not in taxonomy_data:
            raise KeyError(f"Missing key '{key}' in taxonomy file")

    color_labels = set(map(normalize_label, taxonomy_data["COLOR_LABELS"]))
    material_labels = set(map(normalize_label, taxonomy_data["MATERIAL_LABELS"]))
    attribute_labels = set(map(normalize_label, taxonomy_data["ATTRIBUTE_LABELS"]))

    keyword_taxonomy = (
        [normalize_label(x) for x in taxonomy_data["COLOR_LABELS"]]
        + [normalize_label(x) for x in taxonomy_data["MATERIAL_LABELS"]]
        + [normalize_label(x) for x in taxonomy_data["ATTRIBUTE_LABELS"]]
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    print("Building global text matrix...")
    global_text_matrix = build_text_matrix_fast(
        model=model,
        tokenizer=tokenizer,
        keyword_taxonomy=keyword_taxonomy,
        device=device,
        color_labels=color_labels,
        material_labels=material_labels,
        attribute_labels=attribute_labels,
        batch_size=text_batch_size,
        use_amp=use_amp,
        base_segment_label=None,
    )

    records = load_json_records_from_folder(segments_json_dir)
    print(f"Loaded records: {len(records)}")

    # Listen für alle verarbeiteten Segmente. Daraus werden später Matrizen erstellt.
    segment_ids: List[str] = []
    segment_labels: List[str] = []
    segment_semantic_labels: List[str] = []
    segment_image_ids: List[str] = []

    segment_embs: List[np.ndarray] = []
    segment_label_text_embs: List[np.ndarray] = []
    concept_scores: List[np.ndarray] = []
    sparse_concepts: List[List[int]] = []
    segments_out: List[Dict[str, Any]] = []

    conditioned_cache: Dict[str, torch.Tensor] = {}
    label_text_embedding_cache: Dict[str, torch.Tensor] = {}

    image_count = 0
    segment_count_total = 0
    skipped_images = 0
    skipped_segments = 0

    for rec in tqdm(records, desc="Embedding segments + concept projection"):
        image_count += 1

        image_id = str(rec.get("image_id", ""))
        webcam_id = str(rec.get("webcam_id", ""))
        image_uid = str(
            rec.get(
                "image_uid",
                f"{webcam_id}_{image_id}" if webcam_id and image_id else f"image_{image_count}",
            )
        )

        image_file = str(rec.get("image_file", ""))
        image_path_raw = rec.get("image_path", "")
        image_w = int(rec.get("width", 0))
        image_h = int(rec.get("height", 0))
        segmap_file = rec.get("segmentation_map_npy")
        segments = rec.get("segments", [])

        if not segments or not segmap_file:
            continue

        if not image_path_raw:
            print(f"[WARN] Skipping {image_uid}: no image_path in json")
            skipped_images += 1
            continue

        image_path = resolve_path(image_path_raw, images_base_dir)
        segmap_path = resolve_path(segmap_file, segmaps_dir)

        try:
            with Image.open(image_path) as img:
                full_img = img.convert("RGB")
        except Exception as e:
            print(f"[WARN] Skipping image {image_uid} ({image_path}): {e}")
            skipped_images += 1
            continue

        try:
            segmap = load_segmap_array(segmap_path)
        except Exception as e:
            print(f"[WARN] Skipping segmap {image_uid} ({segmap_path}): {e}")
            skipped_images += 1
            continue

        if segmap.ndim != 2:
            print(f"[WARN] Skipping image {image_uid}: segmap is not 2D, got {segmap.shape}")
            skipped_images += 1
            continue

        segment_imgs: List[Image.Image] = []
        segment_meta: List[Dict[str, Any]] = []

        for seg in segments:
            try:
                original_label, refined_label, semantic_label = get_original_and_refined_labels(
                    seg=seg,
                    use_low_confidence_refined=use_low_confidence_refined,
                )
                if not original_label:
                    continue

                if not should_keep_segment(seg, min_area, min_area_ratio, image_w, image_h):
                    continue

                seg_id = int(seg["segment_id"])
                bbox_data = seg.get("bbox", {})
                x1 = int(bbox_data.get("x1", 0))
                y1 = int(bbox_data.get("y1", 0))
                x2 = int(bbox_data.get("x2", 0)) + 1
                y2 = int(bbox_data.get("y2", 0)) + 1

                mask = segmap == seg_id
                mask_area = int(mask.sum())
                if mask_area <= 0:
                    continue

                segment_uid = str(seg.get("segmentUid", seg.get("segment_uid", f"{image_uid}_seg_{seg_id}")))

                seg_img = crop_from_mask(
                    full_img,
                    mask=mask,
                    bbox=(x1, y1, x2, y2),
                    pad=crop_pad,
                    black_background=black_background,
                )

                seg_width, seg_height = seg_img.size
                area = int(seg.get("area", mask_area))
                area_ratio = float(area) / max(1, image_w * image_h)

                segment_imgs.append(seg_img)
                segment_meta.append({
                    "segmentUid": segment_uid,
                    "segmentId": seg_id,
                    "categoryId": int(seg.get("category_id", -1)),
                    "imageId": image_uid,
                    "imageFile": image_file,
                    "imagePath": str(image_path),
                    "segmapPath": str(segmap_path),
                    "webcamId": webcam_id,
                    "label": original_label,
                    "originalLabel": original_label,
                    "refinedLabel": refined_label,
                    "semanticLabel": semantic_label,
                    "score": float(seg.get("score", 0.0)),
                    "wasFused": bool(seg.get("was_fused", False)),
                    "area": area,
                    "maskArea": mask_area,
                    "areaRatio": area_ratio,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": seg_width,
                    "height": seg_height,
                })
            except Exception as e:
                print(f"[WARN] Skipping segment in image {image_uid}: {e}")
                skipped_segments += 1

        if not segment_imgs:
            continue

        try:
            batch_embs_t = embed_pil_images_in_chunks(
                model=model,
                preprocess=preprocess,
                pil_imgs=segment_imgs,
                device=device,
                batch_size=image_batch_size,
                use_amp=use_amp,
            ).to(device)
        except Exception as e:
            print(f"[WARN] Batch embedding failed for image {image_uid}: {e}")
            skipped_images += 1
            continue

        # Semantische Labels nur einmal encoden und danach wiederverwenden.
        new_labels = sorted({
            meta["semanticLabel"]
            for meta in segment_meta
            if meta["semanticLabel"] not in label_text_embedding_cache
        })
        if new_labels:
            label_text_embedding_cache.update(
                embed_unique_labels_fast(
                    model=model,
                    tokenizer=tokenizer,
                    labels=new_labels,
                    device=device,
                    batch_size=text_batch_size,
                    use_amp=use_amp,
                )
            )

        label_text_embs_t = torch.stack(
            [label_text_embedding_cache[meta["semanticLabel"]] for meta in segment_meta],
            dim=0,
        ).to(device)

        label_to_local_indices: DefaultDict[str, List[int]] = defaultdict(list)
        for idx, meta in enumerate(segment_meta):
            label_to_local_indices[meta["label"]].append(idx)

        scores_per_segment: Dict[int, np.ndarray] = {}

        for seg_label, indices in label_to_local_indices.items():
            if use_conditioned_prompts:
                if seg_label not in conditioned_cache:
                    conditioned_cache[seg_label] = build_text_matrix_fast(
                        model=model,
                        tokenizer=tokenizer,
                        keyword_taxonomy=keyword_taxonomy,
                        device=device,
                        color_labels=color_labels,
                        material_labels=material_labels,
                        attribute_labels=attribute_labels,
                        batch_size=text_batch_size,
                        use_amp=use_amp,
                        base_segment_label=seg_label,
                    )
                text_matrix = conditioned_cache[seg_label]
            else:
                text_matrix = global_text_matrix

            idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            emb_group = batch_embs_t.index_select(0, idx_tensor)

            scores_group_t = torch.clamp(emb_group @ text_matrix.T, min=0)
            scores_group = scores_group_t.cpu().numpy().astype(np.float32)

            for local_pos, seg_idx_local in enumerate(indices):
                scores_per_segment[seg_idx_local] = scores_group[local_pos]

        batch_embs = batch_embs_t.cpu().numpy().astype(np.float32)
        label_text_embs = label_text_embs_t.cpu().numpy().astype(np.float32)

        for i, meta in enumerate(segment_meta):
            emb = batch_embs[i]
            label_text_emb = label_text_embs[i]
            scores = scores_per_segment[i]

            kept_idx = select_best_concepts_by_category(
                scores=scores,
                keyword_taxonomy=keyword_taxonomy,
                color_labels=color_labels,
                material_labels=material_labels,
                attribute_labels=attribute_labels,
                concept_threshold=concept_threshold,
                max_colors=1,
                max_materials=1,
                max_attributes=2,
            )
            if max_concepts_per_segment > 0:
                kept_idx = kept_idx[:max_concepts_per_segment]

            top_kw, top_sc = topk_from_scores(scores=scores, labels=keyword_taxonomy, k=top_kw_segment)

            segment_ids.append(meta["segmentUid"])
            segment_labels.append(meta["label"])
            segment_semantic_labels.append(meta["semanticLabel"])
            segment_image_ids.append(meta["imageId"])

            segment_embs.append(emb)
            segment_label_text_embs.append(label_text_emb)
            concept_scores.append(scores)
            sparse_concepts.append(kept_idx)

            segments_out.append({
                **meta,
                "topKw": top_kw,
                "topKwScores": top_sc,
                "keptConcepts": [keyword_taxonomy[j] for j in kept_idx],
                "keptConceptScores": [float(scores[j]) for j in kept_idx],
            })
            segment_count_total += 1

    if not segment_embs:
        print("No segments processed.")
        return

    # Vektoren normalisieren, damit Skalarprodukte Cosine Similarities entsprechen.
    print("Preparing vectors...")
    segment_embs_np = np.vstack(segment_embs).astype(np.float32)
    concept_scores_np = np.vstack(concept_scores).astype(np.float32)
    segment_label_text_embs_np = np.vstack(segment_label_text_embs).astype(np.float32)

    segment_embs_np /= np.linalg.norm(segment_embs_np, axis=1, keepdims=True) + 1e-12
    concept_vecs_np = concept_scores_np / (np.linalg.norm(concept_scores_np, axis=1, keepdims=True) + 1e-12)
    segment_label_text_embs_np /= np.linalg.norm(segment_label_text_embs_np, axis=1, keepdims=True) + 1e-12

    n_segments = segment_embs_np.shape[0]

    label_to_segment_indices: DefaultDict[str, List[int]] = defaultdict(list)
    concept_to_segment_indices: DefaultDict[int, List[int]] = defaultdict(list)

    for idx, label in enumerate(segment_labels):
        label_to_segment_indices[label].append(idx)
        for concept_idx in sparse_concepts[idx]:
            concept_to_segment_indices[concept_idx].append(idx)

    edges_out: List[Dict[str, Any]] = []

    # Für jedes Segment die besten Ähnlichkeitskanten berechnen.
    for i in tqdm(range(n_segments), desc="Building segment graph edges"):
        my_label = segment_labels[i]
        my_image_id = segment_image_ids[i]
        my_concepts = sparse_concepts[i]

        if compare_only_same_label:
            same_label_candidates = label_to_segment_indices[my_label]
        else:
            same_label_candidates = range(n_segments)

        if use_concept_prefilter and my_concepts:
            concept_candidates: Set[int] = set()
            for c in my_concepts:
                # Begrenzung verhindert, dass häufige Konzepte wie "gray" die Laufzeit dominieren.
                concept_candidates.update(concept_to_segment_indices[c][:max_segments_per_concept])

            cand = [
                j for j in concept_candidates
                if j != i
                and (not compare_only_same_label or segment_labels[j] == my_label)
                and (not exclude_same_image or segment_image_ids[j] != my_image_id)
            ]

            # Fallback, falls ein Segment keine sinnvolle Konzeptüberschneidung hat.
            if len(cand) < edges_per_segment:
                cand = [
                    j for j in same_label_candidates
                    if j != i and (not exclude_same_image or segment_image_ids[j] != my_image_id)
                ]
        else:
            cand = [
                j for j in same_label_candidates
                if j != i and (not exclude_same_image or segment_image_ids[j] != my_image_id)
            ]

        if not cand:
            continue

        cand_np = np.asarray(cand, dtype=np.int32)

        # Visueller Vorfilter: finalen Score nur für die stärksten Kandidaten berechnen.
        if cand_np.shape[0] > max_candidates_per_segment:
            pre_sims = segment_embs_np[cand_np] @ segment_embs_np[i]
            top_idx = np.argpartition(-pre_sims, max_candidates_per_segment - 1)[:max_candidates_per_segment]
            top_idx = top_idx[np.argsort(-pre_sims[top_idx])]
            cand_np = cand_np[top_idx]

        sim_segments = segment_embs_np[cand_np] @ segment_embs_np[i]
        sim_concepts = concept_vecs_np[cand_np] @ concept_vecs_np[i]
        sim_refined_labels = segment_label_text_embs_np[cand_np] @ segment_label_text_embs_np[i]

        sim_blends = (
            w_segment * sim_segments
            + w_concept * sim_concepts
            + w_refined_label * sim_refined_labels
        )

        keep_k = min(edges_per_segment, sim_blends.shape[0])
        if keep_k <= 0:
            continue

        best_idx = np.argpartition(-sim_blends, keep_k - 1)[:keep_k]
        best_idx = best_idx[np.argsort(-sim_blends[best_idx])]

        for pos in best_idx:
            j = int(cand_np[pos])

            edges_out.append({
                "srcSegmentUid": segment_ids[i],
                "dstSegmentUid": segment_ids[j],
                "srcLabel": segment_labels[i],
                "dstLabel": segment_labels[j],
                "srcSemanticLabel": segment_semantic_labels[i],
                "dstSemanticLabel": segment_semantic_labels[j],
                "sim": float(sim_blends[pos]),
                "simSegment": float(sim_segments[pos]),
                "simConcept": float(sim_concepts[pos]),
                "simRefinedLabel": float(sim_refined_labels[pos]),
            })

    # Kompakte Outputs für Graph-Import und spätere Auswertung speichern.
    out_segments_path = out_dir_p / "segments_out.json"
    out_edges_path = out_dir_p / "segment_edges_out.json"
    config_path = out_dir_p / "segment_graph_config.json"

    with open(out_segments_path, "w", encoding="utf-8") as f:
        json.dump(segments_out, f, ensure_ascii=False, indent=2)

    with open(out_edges_path, "w", encoding="utf-8") as f:
        json.dump(edges_out, f, ensure_ascii=False, indent=2)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "pretrained": pretrained,
                "device": device,
                "image_batch_size": image_batch_size,
                "text_batch_size": text_batch_size,
                "use_amp": use_amp,
                "top_kw_segment": top_kw_segment,
                "concept_threshold": concept_threshold,
                "max_concepts_per_segment": max_concepts_per_segment,
                "max_segments_per_concept": max_segments_per_concept,
                "max_candidates_per_segment": max_candidates_per_segment,
                "edges_per_segment": edges_per_segment,
                "w_segment": w_segment,
                "w_concept": w_concept,
                "w_refined_label": w_refined_label,
                "use_conditioned_prompts": use_conditioned_prompts,
                "use_concept_prefilter": use_concept_prefilter,
                "exclude_same_image": exclude_same_image,
                "compare_only_same_label": compare_only_same_label,
                "crop_pad": crop_pad,
                "black_background": black_background,
                "min_area": min_area,
                "min_area_ratio": min_area_ratio,
                "use_low_confidence_refined": use_low_confidence_refined,
                "outputs": {
                    "segments": str(out_segments_path),
                    "edges": str(out_edges_path),
                },
                "label_strategy": (
                    "Candidate grouping uses original label, e.g. door->door. "
                    "refinedLabel/displayLabel is used as an additional semantic similarity signal."
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    elapsed = time.perf_counter() - start_time
    print(f"Total runtime: {elapsed:.2f} seconds")
    print(f"Total runtime: {elapsed / 60:.2f} minutes")
    print(f"Total runtime: {elapsed / 3600:.2f} hours")
    print(f"Wrote: {out_segments_path}")
    print(f"Wrote: {out_edges_path}")
    print(f"Wrote: {config_path}")
    print(f"Images read: {image_count}")
    print(f"Segments processed: {segment_count_total}")
    print(f"Keyword concepts: {len(keyword_taxonomy)}")
    print(f"Edges: {len(edges_out)}")
    print(f"Conditioned prompt cache: {len(conditioned_cache)}")
    print(f"Label text embedding cache: {len(label_text_embedding_cache)}")
    print(f"Skipped images: {skipped_images}")
    print(f"Skipped segments: {skipped_segments}")

if __name__ == "__main__":
    main(
        segments_json_dir=SEGMENTS_JSON_DIR,
        segmaps_dir=SEGMAPS_DIR,
        segment_taxonomy_json_path=SEGMENT_TAXONOMY_JSON,
        out_dir=OUT_DIR,
        images_base_dir=IMAGES_BASE_DIR,
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
        device=DEVICE,
        image_batch_size=IMAGE_BATCH_SIZE,
        text_batch_size=TEXT_BATCH_SIZE,
        use_amp=USE_AMP,
        top_kw_segment=TOP_KW_SEGMENT,
        concept_threshold=CONCEPT_THRESHOLD,
        max_concepts_per_segment=MAX_CONCEPTS_PER_SEGMENT,
        max_segments_per_concept=MAX_SEGMENTS_PER_CONCEPT,
        max_candidates_per_segment=MAX_CANDIDATES_PER_SEGMENT,
        edges_per_segment=EDGES_PER_SEGMENT,
        w_segment=W_SEGMENT,
        w_concept=W_CONCEPT,
        w_refined_label=W_REFINED_LABEL,
        use_conditioned_prompts=USE_CONDITIONED_PROMPTS,
        use_concept_prefilter=USE_CONCEPT_PREFILTER,
        exclude_same_image=EXCLUDE_SAME_IMAGE,
        compare_only_same_label=COMPARE_ONLY_SAME_LABEL,
        crop_pad=CROP_PAD,
        black_background=BLACK_BACKGROUND,
        min_area=MIN_AREA,
        min_area_ratio=MIN_AREA_RATIO,
        use_low_confidence_refined=USE_LOW_CONFIDENCE_REFINED,
    )
