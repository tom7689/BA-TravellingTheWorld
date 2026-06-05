import os
import json
import time
import urllib.request
from typing import Any, DefaultDict, Dict, List, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
import open_clip
from torchvision import models, transforms


# -----------------------------
# Konfiguration
# -----------------------------
# Eingabe- und Ausgabepfade
IMAGES_JSON_PATH = r"Scenes\scenes.json"
TAXONOMY_JSON_PATH = r"Scenes\KeywordsScenes.json"
OUT_DIR = r"Scenes\output_scenesEmb1"



PLACES365_WEIGHTS_PATH = r"Scenes\resnet50_places365.pth.tar"
PLACES365_LABELS_PATH = r"Scenes\categories_places365.txt"

# Modellkonfiguration
CLIP_MODEL_NAME = "ViT-B-16"
CLIP_PRETRAINED = "laion2b_s34b_b88k"
USE_PROMPT_ENSEMBLE = True
USE_CUDA_IF_AVAILABLE = True

# Ausgabe der wichtigsten Keywords pro Bild
TOP_KW_CLIP = 2
TOP_KW_PLACES = 4

# Thresholds und maximale Anzahl Konzepte pro Bild
CLIP_THRESHOLD = 0.20
PLACES_THRESHOLD = 0.05
MAX_CLIP_CONCEPTS_PER_IMAGE = 2
MAX_PLACES_CONCEPTS_PER_IMAGE = 4

# Begrenzungen für die Kandidatensuche und die Kanten im Graphen
MAX_IMAGES_PER_CONCEPT = 500
MAX_CANDIDATES_PER_IMAGE = 3500
EDGES_PER_IMAGE = 20
TOP_SHARED_CONCEPTS = 5

# Gewichtung der finalen Ähnlichkeit
W_IMG = 0.60
W_CONCEPT = 0.40

# Download-Quellen für Places365, falls die Dateien lokal noch nicht vorhanden sind
PLACES365_WEIGHTS_URL = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
PLACES365_LABELS_URL = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"


# -----------------------------
# Allgemeine Hilfsfunktionen
# -----------------------------
def to_unit(x: torch.Tensor) -> torch.Tensor:
    """Normalisiert Tensoren auf Einheitslänge für Kosinus-Ähnlichkeiten."""
    return F.normalize(x, dim=-1)


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    """Berechnet die Kosinus-Ähnlichkeit für bereits normalisierte NumPy-Vektoren."""
    return float(np.dot(a, b))


def topk_from_scores(scores: np.ndarray, labels: List[str], k: int) -> Tuple[List[str], List[float]]:
    """Liest die k höchsten Labels und Scores aus einem Score-Vektor aus."""
    if k <= 0:
        return [], []

    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    return [labels[i] for i in idx], [float(scores[i]) for i in idx]


def sparse_top_indices(scores: np.ndarray, threshold: float, max_items: int) -> np.ndarray:
    """Reduziert einen Score-Vektor auf relevante Konzepte oberhalb eines Schwellwerts."""
    idx = np.where(scores >= threshold)[0]

    if idx.size > max_items:
        top_idx = np.argpartition(-scores, kth=max_items - 1)[:max_items]
        return top_idx[np.argsort(-scores[top_idx])]

    return idx[np.argsort(-scores[idx])]


# -----------------------------
# OpenCLIP
# -----------------------------
@torch.no_grad()
def embed_image(model, preprocess, img_path: str, device: str) -> torch.Tensor:
    """Erzeugt ein normalisiertes OpenCLIP-Bildembedding."""
    img = Image.open(img_path).convert("RGB")
    image = preprocess(img).unsqueeze(0).to(device)
    feat = model.encode_image(image)
    return to_unit(feat).squeeze(0)


@torch.no_grad()
def build_text_matrix_single(model, tokenizer, labels: List[str], device: str) -> torch.Tensor:
    """Erzeugt Textembeddings direkt aus den Labelnamen."""
    tokens = tokenizer(labels).to(device)
    text_feat = model.encode_text(tokens)
    return to_unit(text_feat)


def get_templates_for_label(label: str) -> List[str]:
    """Prompt-Vorlagen für die Condition-Keywords."""
    return [
        "a photo of a scene with {}",
        "an outdoor scene with {}",
        "a view showing {}",
        "a scene under {} conditions"
    ]


@torch.no_grad()
def build_text_matrix_ensemble(model, tokenizer, labels: List[str], device: str) -> torch.Tensor:
    """Erzeugt stabilere Textembeddings durch mehrere Prompt-Varianten pro Label."""
    out = []

    for label in labels:
        prompts = [template.format(label) for template in get_templates_for_label(label)]
        tokens = tokenizer(prompts).to(device)

        feat = model.encode_text(tokens)
        feat = F.normalize(feat, dim=-1)
        feat = feat.mean(dim=0)
        feat = F.normalize(feat, dim=-1)
        out.append(feat)

    return torch.stack(out, dim=0)


@torch.no_grad()
def concept_projection(text_matrix: torch.Tensor, img_emb: torch.Tensor) -> torch.Tensor:
    """Projiziert ein Bildembedding auf die Textkonzepte."""
    return text_matrix @ img_emb


# -----------------------------
# Places365
# -----------------------------
def load_places365_model(device: str):
    """Lädt das Places365-ResNet und die passende Bildtransformation."""
    model = models.resnet50(num_classes=365)

    if not os.path.exists(PLACES365_WEIGHTS_PATH):
        urllib.request.urlretrieve(PLACES365_WEIGHTS_URL, PLACES365_WEIGHTS_PATH)

    checkpoint = torch.load(PLACES365_WEIGHTS_PATH, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, tfm


def load_places365_labels() -> List[str]:
    """Lädt die 365 Places-Kategorien und bereitet die Labelnamen auf."""
    if not os.path.exists(PLACES365_LABELS_PATH):
        urllib.request.urlretrieve(PLACES365_LABELS_URL, PLACES365_LABELS_PATH)

    classes = []
    with open(PLACES365_LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            cls = line.strip().split(" ")[0][3:]
            classes.append(cls.replace("_", " "))

    return classes


@torch.no_grad()
def predict_places_scores(places_model, places_transform, img_path: str, device: str) -> np.ndarray:
    """Berechnet die Places365-Wahrscheinlichkeiten für ein Bild."""
    img = Image.open(img_path).convert("RGB")
    x = places_transform(img).unsqueeze(0).to(device)

    logits = places_model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)

    return probs


# -----------------------------
# Feature-Extraktion
# -----------------------------
@torch.no_grad()
def get_image_features(
    img_path: str,
    clip_model,
    clip_preprocess,
    clip_text_matrix: torch.Tensor,
    places_model,
    places_transform,
    device: str,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Berechnet OpenCLIP-Bildembedding, CLIP-Keyword-Scores und Places365-Scores."""
    emb_t = embed_image(clip_model, clip_preprocess, img_path, device)

    clip_scores_t = concept_projection(clip_text_matrix, emb_t)
    clip_scores_t = torch.clamp(clip_scores_t, min=0)
    clip_scores = clip_scores_t.detach().cpu().numpy().astype(np.float32)

    places_scores = predict_places_scores(
        places_model,
        places_transform,
        img_path,
        device,
    )

    return emb_t, clip_scores, places_scores


# -----------------------------
# Hauptpipeline
# -----------------------------
def main():
    # Grundkonfiguration und Eingabedaten
    device = "cuda" if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(IMAGES_JSON_PATH, "r", encoding="utf-8") as f:
        images_in: List[Dict[str, Any]] = json.load(f)

    with open(TAXONOMY_JSON_PATH, "r", encoding="utf-8") as f:
        taxonomy_data = json.load(f)

    clip_taxonomy: List[str] = taxonomy_data["CONDITION_LABELS"]
    places_taxonomy: List[str] = load_places365_labels()
    combined_taxonomy: List[str] = clip_taxonomy + places_taxonomy

    # Modelle laden
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        device=device,
    )
    clip_model.eval()

    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    places_model, places_transform = load_places365_model(device)

    start_time = time.perf_counter()

    if USE_PROMPT_ENSEMBLE:
        clip_text_matrix = build_text_matrix_ensemble(
            clip_model,
            tokenizer,
            clip_taxonomy,
            device,
        )
    else:
        clip_text_matrix = build_text_matrix_single(
            clip_model,
            tokenizer,
            clip_taxonomy,
            device,
        )

    # Bildfeatures und Konzepte extrahieren
    img_ids: List[str] = []
    img_embs: List[np.ndarray] = []
    concept_scores: List[np.ndarray] = []
    sparse_concepts: List[List[int]] = []
    images_out: List[Dict[str, Any]] = []

    for rec in tqdm(images_in, desc="Embedding images + concepts"):
        iid = str(rec["sceneId"])
        path = str(rec["path"])

        with Image.open(path) as img_meta:
            width, height = img_meta.size

        emb_t, clip_scores, places_scores = get_image_features(
            img_path=path,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            clip_text_matrix=clip_text_matrix,
            places_model=places_model,
            places_transform=places_transform,
            device=device,
        )

        emb = emb_t.detach().cpu().numpy().astype(np.float32)

        top_clip_kw, top_clip_sc = topk_from_scores(clip_scores, clip_taxonomy, TOP_KW_CLIP)
        top_places_kw, top_places_sc = topk_from_scores(places_scores, places_taxonomy, TOP_KW_PLACES)

        clip_idx = sparse_top_indices(
            clip_scores,
            threshold=CLIP_THRESHOLD,
            max_items=MAX_CLIP_CONCEPTS_PER_IMAGE,
        )
        places_idx = sparse_top_indices(
            places_scores,
            threshold=PLACES_THRESHOLD,
            max_items=MAX_PLACES_CONCEPTS_PER_IMAGE,
        )

        combined_scores = np.concatenate([clip_scores, places_scores], axis=0)
        places_idx_shifted = places_idx + len(clip_taxonomy)
        combined_sparse_idx = np.concatenate([clip_idx, places_idx_shifted], axis=0).tolist()

        img_ids.append(iid)
        img_embs.append(emb)
        concept_scores.append(combined_scores)
        sparse_concepts.append(combined_sparse_idx)

        images_out.append({
            "id": iid,
            "path": path,
            "width": width,
            "height": height,
            "topClipKw": top_clip_kw,
            "topClipScores": top_clip_sc,
            "topPlacesKw": top_places_kw,
            "topPlacesScores": top_places_sc,
        })

    # Normalisierung und Konzeptindex für Kandidatensuche
    img_embs = np.vstack(img_embs)
    concept_scores = np.vstack(concept_scores)
    n_images, _ = img_embs.shape

    img_embs /= np.linalg.norm(img_embs, axis=1, keepdims=True) + 1e-12

    concept_vecs = concept_scores.copy()
    concept_vecs /= np.linalg.norm(concept_vecs, axis=1, keepdims=True) + 1e-12

    concept_to_images_scored: DefaultDict[int, List[Tuple[int, float]]] = defaultdict(list)

    for i in range(n_images):
        for concept_id in sparse_concepts[i]:
            concept_to_images_scored[concept_id].append((i, float(concept_scores[i, concept_id])))

    concept_to_images: DefaultDict[int, List[int]] = defaultdict(list)

    for concept_id, matches in concept_to_images_scored.items():
        matches.sort(key=lambda x: x[1], reverse=True)
        concept_to_images[concept_id] = [i for i, _score in matches[:MAX_IMAGES_PER_CONCEPT]]

    # Ähnlichkeitskanten berechnen
    edges_out: List[Dict[str, Any]] = []

    for i in tqdm(range(n_images), desc="Building graph edges"):
        candidates: Dict[int, float] = {}
        my_concepts = sparse_concepts[i]
        my_concepts_set = set(my_concepts)

        for concept_id in my_concepts:
            score_a = float(concept_scores[i, concept_id])

            for j in concept_to_images[concept_id]:
                if j == i:
                    continue

                score_b = float(concept_scores[j, concept_id])
                candidates[j] = candidates.get(j, 0.0) + min(score_a, score_b)

        if not candidates:
            continue

        if len(candidates) > MAX_CANDIDATES_PER_IMAGE:
            candidates = dict(
                sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:MAX_CANDIDATES_PER_IMAGE]
            )

        scored: List[Tuple[int, float, float, float]] = []
        emb_a = img_embs[i]
        concept_vec_a = concept_vecs[i]

        for j in candidates.keys():
            sim_img = cosine_np(emb_a, img_embs[j])
            sim_concept = cosine_np(concept_vec_a, concept_vecs[j])
            sim_blend = (W_IMG * sim_img) + (W_CONCEPT * sim_concept)

            scored.append((j, sim_blend, sim_img, sim_concept))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:EDGES_PER_IMAGE]

        for j, sim_blend, sim_img, sim_concept in scored:
            shared = my_concepts_set.intersection(sparse_concepts[j])

            if shared:
                shared_list = sorted(
                    [
                        (concept_id, min(float(concept_scores[i, concept_id]), float(concept_scores[j, concept_id])))
                        for concept_id in shared
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )[:TOP_SHARED_CONCEPTS]

                shared_names = [combined_taxonomy[concept_id] for concept_id, _ in shared_list]
                shared_scores = [float(score) for _, score in shared_list]
            else:
                shared_names, shared_scores = [], []

            edges_out.append({
                "src": img_ids[i],
                "dst": img_ids[j],
                "sim": float(sim_blend),
                "simImg": float(sim_img),
                "simConcept": float(sim_concept),
                "sharedConcepts": shared_names,
                "sharedConceptScores": shared_scores,
            })

    # Ausgabe speichern
    out_images_path = os.path.join(OUT_DIR, "images_out.json")
    out_edges_path = os.path.join(OUT_DIR, "edges_out.json")

    with open(out_images_path, "w", encoding="utf-8") as f:
        json.dump(images_out, f, ensure_ascii=False, indent=2)

    with open(out_edges_path, "w", encoding="utf-8") as f:
        json.dump(edges_out, f, ensure_ascii=False, indent=2)

    # Laufzeit und Übersicht ausgeben
    elapsed = time.perf_counter() - start_time

    print(f"⏱️ Total runtime: {elapsed:.2f} seconds")
    print(f"⏱️ Total runtime: {elapsed / 60:.2f} minutes")
    print(f"⏱️ Total runtime: {elapsed / 3600:.2f} hours")
    print("✅ Wrote:", out_images_path)
    print("✅ Wrote:", out_edges_path)
    print(f"✅ Images: {n_images}")
    print(f"✅ CLIP concepts: {len(clip_taxonomy)}")
    print(f"✅ Places concepts: {len(places_taxonomy)}")
    print(f"✅ Combined concepts: {len(combined_taxonomy)}")
    print(f"✅ Edges: {len(edges_out)}")


# -----------------------------
# Skriptstart
# -----------------------------
if __name__ == "__main__":
    main()
