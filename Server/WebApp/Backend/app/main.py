import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from neo4j import GraphDatabase
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image

app = FastAPI()
# uvicorn app.main:app --reload

PROJECT_ROOT_ENV = os.getenv("PROJECT_ROOT")

if PROJECT_ROOT_ENV:
    PROJECT_ROOT = Path(PROJECT_ROOT_ENV).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGES_ROOT = Path(os.getenv("IMAGES_ROOT", PROJECT_ROOT / "ImagesAll2")).resolve()

app.mount("/images", StaticFiles(directory=str(IMAGES_ROOT)), name="images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://160.85.67.175:8080",
        "http://localhost:5173",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


driver_main = GraphDatabase.driver(
    os.getenv("NEO4J_URI_MAIN"),
    auth=(
        os.getenv("NEO4J_USER_MAIN"),
        os.getenv("NEO4J_PASSWORD_MAIN")
    )
)

driver_efficient = GraphDatabase.driver(
    os.getenv("NEO4J_URI_EFFICIENT"),
    auth=(
        os.getenv("NEO4J_USER_EFFICIENT"),
        os.getenv("NEO4J_PASSWORD_EFFICIENT")
    )
)

driver_efficient_refined = GraphDatabase.driver(
    os.getenv("NEO4J_URI_EFFICIENT_REFINED"),
    auth=(
        os.getenv("NEO4J_USER_EFFICIENT_REFINED"),
        os.getenv("NEO4J_PASSWORD_EFFICIENT_REFINED")
    )
)


VALID_DBS = {
    "main": driver_main,
    "efficient": driver_efficient,
    "efficient_refined": driver_efficient_refined,
}


def get_driver(db: str):
    if db not in VALID_DBS:
        raise HTTPException(status_code=400, detail=f"Unknown db variant: {db}")

    return VALID_DBS[db]

def normalize_scores(scores):
    if not scores:
        return []

    mn = min(scores)
    mx = max(scores)

    if mx == mn:
        return [1.0 for _ in scores]

    return [(s - mn) / (mx - mn) for s in scores]


def path_to_image_url(path_str: str) -> str:
    p = Path(path_str)
    rel = f"{p.parent.name}/{p.name}"
    return f"/images/{rel}"


def segment_to_dict(s):
    return {
        "segmentUid": s.get("id") or s.get("segmentUid"),
        "segmentId": s.get("segmentId"),
        "label": s.get("label"),

        # refined labels for frontend hover text
        "originalLabel": s.get("originalLabel") or s.get("label"),
        "refinedLabel": s.get("refinedLabel"),
        "displayLabel": s.get("displayLabel") or s.get("refinedLabel") or s.get("label"),

        "score": s.get("score"),
        "categoryId": s.get("categoryId"),
        "area": s.get("area"),
        "areaRatio": s.get("areaRatio"),
        "maskArea": s.get("maskArea"),
        "x1": s.get("x1"),
        "y1": s.get("y1"),
        "x2": s.get("x2"),
        "y2": s.get("y2"),
        "width": s.get("width"),
        "height": s.get("height"),
        "wasFused": s.get("wasFused"),
        "segmapPath": s.get("segmapPath"),
        "topKw": s.get("topKw", []) or [],
        "topKwScores": s.get("topKwScores", []) or [],
    }


def image_record_to_dict(r, *, include_score=False, extra=None):
    data = {
        "imageUid": r["imageUid"],
        "imageUrl": path_to_image_url(r["path"]),
        "width": r["width"],
        "height": r["height"],
        "webcamId": r.get("webcamId"),
        "latitude": r.get("latitude"),
        "longitude": r.get("longitude"),
        "city": r.get("city"),
        "region": r.get("region"),
        "countryName": r.get("countryName"),
    }

    if include_score:
        data["score"] = float(r["score"]) if r["score"] is not None else None

    if extra:
        data.update(extra)

    return data


def load_segmap(segmap_file: Path) -> np.ndarray:
    if segmap_file.exists():
        if segmap_file.suffix.lower() == ".npz":
            with np.load(segmap_file) as data:
                return data["segmap"]
        return np.load(segmap_file)

    raise FileNotFoundError(f"Segmap not found: {segmap_file}")


SEGMENT_MAP_CYPHER = """
{
    id: s.id,
    segmentId: s.segmentId,
    label: s.label,

    originalLabel: coalesce(s.originalLabel, s.label),
    refinedLabel: s.refinedLabel,
    displayLabel: coalesce(s.displayLabel, s.refinedLabel, s.label),

    score: s.score,
    categoryId: s.categoryId,
    area: s.area,
    areaRatio: s.areaRatio,
    maskArea: s.maskArea,
    x1: s.x1,
    y1: s.y1,
    x2: s.x2,
    y2: s.y2,
    width: s.width,
    height: s.height,
    wasFused: s.wasFused,
    segmapPath: s.segmapPath,
    topKw: s.topKw,
    topKwScores: s.topKwScores
}
"""


@app.get("/api/images/random")
def random_images(
    db: str = Query("main"),
    n: int = Query(5, ge=1, le=50),
):
    query = f"""
        MATCH (i:Image)
        WITH split(i.id, "_")[0] AS cam, collect(i) AS imgs
        WITH cam, imgs[toInteger(rand() * size(imgs))] AS picked

        OPTIONAL MATCH (picked)-[:HAS_SEGMENT]->(s:Segment)
        WITH picked, collect(s) AS segmentNodes

        RETURN picked.id AS imageUid,
            picked.path AS path,
            picked.width AS width,
            picked.height AS height,
            picked.topPlacesKw AS topPlacesKw,
            picked.topClipKw AS topClipKw,
            picked.webcamId AS webcamId,
            picked.latitude AS latitude,
            picked.longitude AS longitude,
            picked.city AS city,
            picked.region AS region,
            picked.countryName AS countryName,

            [s IN segmentNodes WHERE s IS NOT NULL | {SEGMENT_MAP_CYPHER}] AS segments

        ORDER BY rand()
        LIMIT $n
    """

    with get_driver(db).session(database="neo4j") as session:
        result = session.run(query, n=n)
        images = []

        for r in result:
            segments = [segment_to_dict(s) for s in (r.get("segments", []) or [])]

            images.append(
                image_record_to_dict(
                    r,
                    extra={
                        "topPlacesKw": r.get("topPlacesKw", []) or [],
                        "topClipKw": r.get("topClipKw", []) or [],
                        "segments": segments,
                        # temporary compatibility field if the frontend still expects it
                        "crops": [],
                    },
                )
            )

        return {"images": images}


@app.get("/api/images/similar")
def similar_images(
    imageUid: str,
    db: str = Query("main"),
    n: int = Query(5, ge=1, le=50),
    k: int = Query(10, ge=1, le=50),
    includeSameCamera: bool = Query(False),
    excludeCameras: list[str] = Query(default=[]),
):
    query = f"""
        MATCH (a:Image {{id:$imageUid}})-[r:SIMILAR_TO]-(b:Image)
        WHERE b.id <> a.id
            AND ($includeSameCamera = true OR split(b.id, "_")[0] <> split(a.id, "_")[0])
            AND NOT split(b.id, "_")[0] IN $excludeCameras

        WITH split(b.id, "_")[0] AS cam, b, r
        ORDER BY cam, r.sim DESC
        WITH cam, collect({{b:b, sim:r.sim, kw:r.kw, kwScores:r.kwScores}}) AS xs
        WITH cam, xs[0] AS pick
        WHERE pick IS NOT NULL
        WITH pick, pick.b AS b

        OPTIONAL MATCH (b)-[:HAS_SEGMENT]->(s:Segment)
        WITH pick, b, collect(s) AS segmentNodes

        RETURN b.id AS imageUid,
            b.path AS path,
            b.width AS width,
            b.height AS height,
            b.webcamId AS webcamId,
            b.latitude AS latitude,
            b.longitude AS longitude,
            b.city AS city,
            b.region AS region,
            b.countryName AS countryName,
            pick.sim AS score,
            pick.kw AS kw,
            pick.kwScores AS kwScores,

            [s IN segmentNodes WHERE s IS NOT NULL | {SEGMENT_MAP_CYPHER}] AS segments

        ORDER BY score DESC
        LIMIT $n
    """

    with get_driver(db).session(database="neo4j") as session:
        result = session.run(
            query,
            imageUid=imageUid,
            n=n,
            k=k,
            includeSameCamera=includeSameCamera,
            excludeCameras=excludeCameras,
        )

        images = []
        for r in result:
            raw_scores = r.get("kwScores", []) or []
            norm_scores = normalize_scores(raw_scores)
            segments = [segment_to_dict(s) for s in (r.get("segments", []) or [])]

            images.append(
                image_record_to_dict(
                    r,
                    include_score=True,
                    extra={
                        "kw": r.get("kw", []) or [],
                        "kwScores": norm_scores,
                        "kwScoresRaw": raw_scores,
                        "segments": segments,
                        # temporary compatibility field if the frontend still expects it
                        "crops": [],
                    },
                )
            )

        return {"images": images}


@app.get("/api/segments/similar")
def similar_segments(
    segmentUid: str,
    n: int = Query(5, ge=1, le=50),
    includeSameCamera: bool = Query(False),
    excludeImageUid: str | None = Query(None),
    excludeCameras: list[str] = Query(default=[]),
    db: str = Query("main"),
):
    query = f"""
        MATCH (src:Segment {{id:$segmentUid}})<-[:HAS_SEGMENT]-(srcImg:Image)
        MATCH (src)-[r:SIMILAR_TO]-(dst:Segment)<-[:HAS_SEGMENT]-(img:Image)
        WHERE img.id <> srcImg.id
          AND ($excludeImageUid IS NULL OR img.id <> $excludeImageUid)
          AND dst.id <> src.id
          AND dst.label = src.label
          AND ($includeSameCamera = true OR split(img.id, "_")[0] <> split(srcImg.id, "_")[0])
          AND NOT split(img.id, "_")[0] IN $excludeCameras

        WITH img, dst, r
        ORDER BY r.sim DESC

        WITH img, collect({{segment: dst, score: r.sim}}) AS matches
        WITH img, matches[0] AS bestMatch
        WHERE bestMatch IS NOT NULL

        OPTIONAL MATCH (img)-[:HAS_SEGMENT]->(s:Segment)
        WITH img, bestMatch, collect(s) AS allSegmentNodes

        RETURN img.id AS imageUid,
            img.path AS path,
            img.width AS width,
            img.height AS height,
            img.webcamId AS webcamId,
            img.latitude AS latitude,
            img.longitude AS longitude,
            img.city AS city,
            img.region AS region,
            img.countryName AS countryName,
            bestMatch.score AS score,

            {{
                segmentUid: bestMatch.segment.id,
                segmentId: bestMatch.segment.segmentId,
                label: bestMatch.segment.label,

                originalLabel: coalesce(bestMatch.segment.originalLabel, bestMatch.segment.label),
                refinedLabel: bestMatch.segment.refinedLabel,
                displayLabel: coalesce(bestMatch.segment.displayLabel, bestMatch.segment.refinedLabel, bestMatch.segment.label),

                score: bestMatch.segment.score,
                categoryId: bestMatch.segment.categoryId,
                area: bestMatch.segment.area,
                areaRatio: bestMatch.segment.areaRatio,
                maskArea: bestMatch.segment.maskArea,
                x1: bestMatch.segment.x1,
                y1: bestMatch.segment.y1,
                x2: bestMatch.segment.x2,
                y2: bestMatch.segment.y2,
                width: bestMatch.segment.width,
                height: bestMatch.segment.height,
                wasFused: bestMatch.segment.wasFused,
                segmapPath: bestMatch.segment.segmapPath,
                topKw: bestMatch.segment.topKw,
                topKwScores: bestMatch.segment.topKwScores
            }} AS matchedSegment,

            [s IN allSegmentNodes WHERE s IS NOT NULL | {SEGMENT_MAP_CYPHER}] AS segments

        ORDER BY score DESC
        LIMIT $n
    """

    with get_driver(db).session(database="neo4j") as session:
        result = session.run(
            query,
            segmentUid=segmentUid,
            n=n,
            includeSameCamera=includeSameCamera,
            excludeImageUid=excludeImageUid,
            excludeCameras=excludeCameras,
        )

        images = []
        for r in result:
            matched_segment = r["matchedSegment"]

            images.append(
                image_record_to_dict(
                    r,
                    include_score=True,
                    extra={
                        "kw": matched_segment.get("topKw", []) if matched_segment else [],
                        "kwScores": matched_segment.get("topKwScores", []) if matched_segment else [],
                        "sceneKw": r.get("sceneKw", []) or [],
                        "sceneKwScores": r.get("sceneKwScores", []) or [],
                        "matchedSegment": matched_segment,
                        "segments": [segment_to_dict(s) for s in (r.get("segments", []) or [])],
                        # temporary compatibility field if the frontend still expects it
                        "crops": [],
                    },
                )
            )

        return {"images": images}


@app.get("/api/segments/segmap-png")
def get_segmap_png(imageUid: str = Query(...), db: str = Query("main")):
    query = """
        MATCH (img:Image {id:$imageUid})-[:HAS_SEGMENT]->(s:Segment)
        RETURN s.segmapPath AS segmapPath
        LIMIT 1
    """

    with get_driver(db).session(database="neo4j") as session:
        record = session.run(query, imageUid=imageUid).single()

    if record is None:
        raise HTTPException(status_code=404, detail=f"No segmap found for image: {imageUid}")

    segmap_path = record["segmapPath"]

    if not segmap_path:
        raise HTTPException(status_code=400, detail="No segmapPath found")

    segmap_file = Path(str(segmap_path).replace("\\", "/"))
    if not segmap_file.is_absolute():
        segmap_file = (PROJECT_ROOT / segmap_file).resolve()

    try:
        segmap = load_segmap(segmap_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load segmap: {e}")

    if segmap.ndim != 2:
        raise HTTPException(status_code=500, detail="Expected 2D segmap")

    segmap = segmap.astype(np.int32)

    h, w = segmap.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    valid = segmap >= 0
    ids = segmap[valid]

    rgb[..., 0][valid] = (ids & 255).astype(np.uint8)
    rgb[..., 1][valid] = ((ids >> 8) & 255).astype(np.uint8)
    rgb[..., 2][valid] = ((ids >> 16) & 255).astype(np.uint8)

    image = Image.fromarray(rgb, mode="RGB")
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)

    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/images/nearest")
def nearest_image(
    lat: float = Query(...),
    lng: float = Query(...),
    db: str = Query("main"),
):
    query = f"""
        MATCH (i:Image)
        WHERE i.latitude IS NOT NULL
          AND i.longitude IS NOT NULL

        WITH i,
            point({{latitude: $lat, longitude: $lng}}) AS clickPoint,
            point({{latitude: i.latitude, longitude: i.longitude}}) AS imagePoint

        WITH i, point.distance(clickPoint, imagePoint) AS distanceMeters
        ORDER BY distanceMeters ASC
        LIMIT 1

        OPTIONAL MATCH (i)-[:HAS_SEGMENT]->(s:Segment)
        WITH i, distanceMeters, collect(s) AS segmentNodes

        RETURN i.id AS imageUid,
            i.path AS path,
            i.width AS width,
            i.height AS height,
            i.webcamId AS webcamId,
            i.latitude AS latitude,
            i.longitude AS longitude,
            i.city AS city,
            i.region AS region,
            i.countryName AS countryName,
            distanceMeters AS distanceMeters,

            [s IN segmentNodes WHERE s IS NOT NULL | {SEGMENT_MAP_CYPHER}] AS segments
    """

    with get_driver(db).session(database="neo4j") as session:
        r = session.run(query, lat=lat, lng=lng).single()

    if r is None:
        raise HTTPException(status_code=404, detail="No image with coordinates found")

    segments = [segment_to_dict(s) for s in (r.get("segments", []) or [])]

    image = image_record_to_dict(
        r,
        extra={
            "distanceMeters": r.get("distanceMeters"),
            "segments": segments,
            # temporary compatibility field if the frontend still expects it
            "crops": [],
        },
    )

    return {"image": image}
