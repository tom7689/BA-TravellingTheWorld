import json
from pathlib import Path
from tqdm import tqdm


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
IMAGES_DIR = Path(r"ImagesAll2").resolve()
OUT_FILE = Path(r"Scenes\scenes.json").resolve()

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(IMAGES_DIR.rglob("*"))

    scenes = []
    seen_ids = set()

    for p in tqdm(all_files, desc="Scanning images"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in EXTS:
            continue

        if p.name.startswith("."):
            continue

        cam = p.parent.name
        scene_id = f"{cam}_{p.stem}"

        if scene_id in seen_ids:
            print(f"Warning: duplicate sceneId skipped: {scene_id}")
            continue

        seen_ids.add(scene_id)

        scenes.append({
            "sceneId": scene_id,
            "path": p.as_posix()
        })

    OUT_FILE.write_text(
        json.dumps(scenes, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Wrote {len(scenes)} scenes to {OUT_FILE}")


if __name__ == "__main__":
    main()
