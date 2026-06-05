import json
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "123456789"

# clear db:
# MATCH (n)
# DETACH DELETE n;

BATCH_SIZE = 2000
EDGE_BATCH_SIZE = 5000
SEGMENT_BATCH_SIZE = 2000
SEGMENT_SIM_BATCH_SIZE = 5000

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def safe_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value, default=False):
    if value is None:
        return default
    return bool(value)


def safe_str_list(value):
    if not value:
        return []
    return [str(x) for x in value]


def safe_float_list(value):
    if not value:
        return []

    out = []
    for x in value:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            continue

    return out


def create_constraints():
    queries = [
        "CREATE CONSTRAINT image_id IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE",
        "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.id IS UNIQUE",
    ]

    with driver.session(database="neo4j") as session:
        for q in queries:
            session.run(q)

    print("✅ Constraints ensured")


def extract_webcam_id(image_id: str):
    if image_id is None:
        return None

    image_id = str(image_id).strip()
    if not image_id:
        return None

    # erwartet z.B. "1511_img123"
    parts = image_id.split("_")
    if not parts:
        return None

    try:
        return int(parts[0])
    except ValueError:
        return None


# -------------------------------------------------
# IMAGE IMPORT
# -------------------------------------------------

def import_images(path):
    with open(path, "r", encoding="utf-8") as f:
        images = json.load(f)

    cleaned = []
    skipped = 0

    for r in images:
        iid = r.get("id")

        if iid is None or str(iid).strip() == "":
            skipped += 1
            continue

        cleaned.append({
            "id": str(iid).strip(),
            "path": r.get("path"),
            "topPlacesKw": safe_str_list(r.get("topPlacesKw", [])),
            "topClipKw": safe_str_list(r.get("topClipKw", [])),
            "width": safe_int(r.get("width")),
            "height": safe_int(r.get("height")),
            "webcamId": extract_webcam_id(iid),
        })

    query = """
    UNWIND $rows AS row
    MERGE (i:Image {id: row.id})
    SET i.path = row.path,
        i.topPlacesKw = row.topPlacesKw,
        i.topClipKw = row.topClipKw,
        i.width = row.width,
        i.height = row.height,
        i.webcamId = row.webcamId
    """

    total = 0

    with driver.session(database="neo4j") as session:
        for batch in chunks(cleaned, BATCH_SIZE):
            session.run(query, rows=batch).consume()
            total += len(batch)

    print(f"✅ Imported images: {total} (skipped {skipped} with missing id)")


def import_image_locations(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    cleaned = []
    skipped = 0

    for r in rows:
        cam_id = r.get("camId")
        lat = r.get("latitude")
        lng = r.get("longitude")

        if cam_id is None or lat is None or lng is None:
            skipped += 1
            continue

        try:
            cleaned.append({
                "camId": int(cam_id),
                "latitude": float(lat),
                "longitude": float(lng),
                "city": r.get("city"),
                "region": r.get("region"),
                "countryName": r.get("country_name"),
            })
        except (TypeError, ValueError):
            skipped += 1

    query = """
    UNWIND $rows AS row
    MATCH (i:Image)
    WHERE i.webcamId = row.camId
    SET i.latitude = row.latitude,
        i.longitude = row.longitude,
        i.city = row.city,
        i.region = row.region,
        i.countryName = row.countryName
    """

    total = 0

    with driver.session(database="neo4j") as session:
        for batch in chunks(cleaned, BATCH_SIZE):
            session.run(query, rows=batch).consume()
            total += len(batch)

    print(f"✅ Imported image locations for {total} camera rows (skipped {skipped})")


def deduplicate_image_edges(edges):
    best = {}

    for r in edges:
        src = r.get("src")
        dst = r.get("dst")

        if src is None or dst is None:
            continue

        key = (str(src).strip(), str(dst).strip())

        if not key[0] or not key[1]:
            continue

        candidate_kw = r.get("sharedConcepts", [])
        candidate_len = len(candidate_kw)

        if key not in best:
            best[key] = r
        else:
            best_len = len(best[key].get("sharedConcepts", []))
            if candidate_len > best_len:
                best[key] = r

    return list(best.values())


def import_image_edges(path):
    with open(path, "r", encoding="utf-8") as f:
        edges = json.load(f)

    edges = deduplicate_image_edges(edges)

    cleaned = []
    skipped = 0

    for r in edges:
        src = r.get("src")
        dst = r.get("dst")

        if src is None or dst is None or str(src).strip() == "" or str(dst).strip() == "":
            skipped += 1
            continue

        cleaned.append({
            "src": str(src).strip(),
            "dst": str(dst).strip(),
            "sim": safe_float(r.get("sim"), 0.0),
            "kw": safe_str_list(r.get("sharedConcepts", [])),
            "kwScores": safe_float_list(r.get("sharedConceptScores", [])),
        })

    non_empty = sum(1 for x in cleaned if len(x["kw"]) > 0)
    print(f"Image edges total: {len(cleaned)}, with non-empty kw: {non_empty}")

    query = """
    UNWIND $rows AS row
    MATCH (a:Image {id: row.src})
    MATCH (b:Image {id: row.dst})
    MERGE (a)-[rel:SIMILAR_TO]->(b)
    SET rel.sim = row.sim,
        rel.kw = row.kw,
        rel.kwScores = row.kwScores
    """

    total = 0

    with driver.session(database="neo4j") as session:
        for batch in chunks(cleaned, EDGE_BATCH_SIZE):
            session.run(query, rows=batch).consume()
            total += len(batch)

    print(f"✅ Imported image edges: {total} (skipped {skipped} with missing src/dst)")



# -------------------------------------------------
# SEGMENT IMPORT
# -------------------------------------------------

def import_segments(path):
    with open(path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    cleaned = []
    skipped = 0

    for obj in segments:
        image_id = obj.get("imageId")
        segment_id = obj.get("segmentUid")

        if image_id is None or str(image_id).strip() == "":
            skipped += 1
            continue

        if segment_id is None or str(segment_id).strip() == "":
            skipped += 1
            continue

        cleaned.append({
            "image_id": str(image_id).strip(),
            "segment_id": str(segment_id).strip(),

            "segmentId": safe_int(obj.get("segmentId")),
            "categoryId": safe_int(obj.get("categoryId"), -1),

            # Hauptlabel bleibt original Mask2Former Label, z. B. door
            "label": obj.get("label"),

            # refined-label Felder
            "originalLabel": obj.get("originalLabel"),
            "refinedLabel": obj.get("refinedLabel"),
            "displayLabel": obj.get("displayLabel"),
            "semanticLabel": obj.get("semanticLabel"),

            # zusätzliche Debug-/Confidence-Felder aus refined JSON / segments_out
            "refinedLabelScore": safe_float(obj.get("refinedLabelScore")),
            "refinementConfidence": obj.get("refinementConfidence"),
            "refinementReason": obj.get("refinementReason"),

            "score": safe_float(obj.get("score"), 0.0),
            "wasFused": safe_bool(obj.get("wasFused"), False),

            "area": safe_int(obj.get("area")),
            "maskArea": safe_int(obj.get("maskArea")),
            "areaRatio": safe_float(obj.get("areaRatio"), 0.0),

            "x1": safe_int(obj.get("x1")),
            "y1": safe_int(obj.get("y1")),
            "x2": safe_int(obj.get("x2")),
            "y2": safe_int(obj.get("y2")),

            "width": safe_int(obj.get("width")),
            "height": safe_int(obj.get("height")),

            "topKw": safe_str_list(obj.get("topKw", [])),
            "topKwScores": safe_float_list(obj.get("topKwScores", [])),

            "keptConcepts": safe_str_list(obj.get("keptConcepts", [])),
            "keptConceptScores": safe_float_list(obj.get("keptConceptScores", [])),

            "segmapPath": obj.get("segmapPath"),
            "imagePath": obj.get("imagePath"),
            "imageFile": obj.get("imageFile"),
            "webcamId": obj.get("webcamId"),
        })

    print(f"Segments prepared: {len(cleaned)}")
    print(f"Skipped segment entries: {skipped}")

    query = """
    UNWIND $rows AS row
    MATCH (img:Image {id: row.image_id})
    MERGE (s:Segment {id: row.segment_id})
    SET s.segmentId = row.segmentId,
        s.categoryId = row.categoryId,

        s.label = row.label,
        s.originalLabel = row.originalLabel,
        s.refinedLabel = row.refinedLabel,
        s.displayLabel = row.displayLabel,
        s.semanticLabel = row.semanticLabel,

        s.refinedLabelScore = row.refinedLabelScore,
        s.refinementConfidence = row.refinementConfidence,
        s.refinementReason = row.refinementReason,

        s.score = row.score,
        s.wasFused = row.wasFused,
        s.area = row.area,
        s.maskArea = row.maskArea,
        s.areaRatio = row.areaRatio,

        s.x1 = row.x1,
        s.y1 = row.y1,
        s.x2 = row.x2,
        s.y2 = row.y2,
        s.width = row.width,
        s.height = row.height,

        s.topKw = row.topKw,
        s.topKwScores = row.topKwScores,
        s.keptConcepts = row.keptConcepts,
        s.keptConceptScores = row.keptConceptScores,

        s.segmapPath = row.segmapPath,
        s.imagePath = row.imagePath,
        s.imageFile = row.imageFile,
        s.webcamId = row.webcamId
    MERGE (img)-[:HAS_SEGMENT]->(s)
    """

    total = 0

    with driver.session(database="neo4j") as session:
        for batch in chunks(cleaned, SEGMENT_BATCH_SIZE):
            session.run(query, rows=batch).consume()
            total += len(batch)

    print(f"✅ Imported segments: {total}")


def deduplicate_segment_similarity_bidirectional(edges):
    best = {}

    for r in edges:
        src = r.get("srcSegmentUid")
        dst = r.get("dstSegmentUid")

        if src is None or dst is None:
            continue

        src = str(src).strip()
        dst = str(dst).strip()

        if not src or not dst or src == dst:
            continue

        sim = safe_float(r.get("sim"), 0.0)

        a, b = sorted([src, dst])
        key = (a, b)

        candidate = {
            "src": a,
            "dst": b,

            "sim": sim,
            "simSegment": safe_float(r.get("simSegment"), 0.0),
            "simConcept": safe_float(r.get("simConcept"), 0.0),
            "simRefinedLabel": safe_float(r.get("simRefinedLabel"), 0.0),

            "sharedStrength": safe_float(r.get("sharedStrength"), 0.0),
            "sharedConcepts": safe_str_list(r.get("sharedConcepts", [])),
            "sharedConceptScores": safe_float_list(r.get("sharedConceptScores", [])),

            "srcLabel": r.get("srcLabel"),
            "dstLabel": r.get("dstLabel"),

            "srcOriginalLabel": r.get("srcOriginalLabel"),
            "dstOriginalLabel": r.get("dstOriginalLabel"),

            "srcRefinedLabel": r.get("srcRefinedLabel"),
            "dstRefinedLabel": r.get("dstRefinedLabel"),

            "srcDisplayLabel": r.get("srcDisplayLabel"),
            "dstDisplayLabel": r.get("dstDisplayLabel"),

            "srcSemanticLabel": r.get("srcSemanticLabel"),
            "dstSemanticLabel": r.get("dstSemanticLabel"),
        }

        if key not in best or sim > safe_float(best[key]["sim"], 0.0):
            best[key] = candidate

    return list(best.values())


def import_segment_similarity(path):
    with open(path, "r", encoding="utf-8") as f:
        edges = json.load(f)

    edges = deduplicate_segment_similarity_bidirectional(edges)

    cleaned = []
    skipped = 0

    for r in edges:
        src = r.get("src")
        dst = r.get("dst")

        if src is None or dst is None or str(src).strip() == "" or str(dst).strip() == "":
            skipped += 1
            continue

        cleaned.append({
            "src": str(src).strip(),
            "dst": str(dst).strip(),

            "sim": safe_float(r.get("sim"), 0.0),
            "simSegment": safe_float(r.get("simSegment"), 0.0),
            "simConcept": safe_float(r.get("simConcept"), 0.0),
            "simRefinedLabel": safe_float(r.get("simRefinedLabel"), 0.0),

            "sharedStrength": safe_float(r.get("sharedStrength"), 0.0),
            "sharedConcepts": safe_str_list(r.get("sharedConcepts", [])),
            "sharedConceptScores": safe_float_list(r.get("sharedConceptScores", [])),

            "srcLabel": r.get("srcLabel"),
            "dstLabel": r.get("dstLabel"),

            "srcOriginalLabel": r.get("srcOriginalLabel"),
            "dstOriginalLabel": r.get("dstOriginalLabel"),

            "srcRefinedLabel": r.get("srcRefinedLabel"),
            "dstRefinedLabel": r.get("dstRefinedLabel"),

            "srcDisplayLabel": r.get("srcDisplayLabel"),
            "dstDisplayLabel": r.get("dstDisplayLabel"),

            "srcSemanticLabel": r.get("srcSemanticLabel"),
            "dstSemanticLabel": r.get("dstSemanticLabel"),
        })

    print(f"Segment similarity edges after deduplication: {len(cleaned)}")

    query = """
    UNWIND $rows AS row
    MATCH (a:Segment {id: row.src})
    MATCH (b:Segment {id: row.dst})
    MERGE (a)-[rel:SIMILAR_TO]->(b)
    SET rel.sim = row.sim,
        rel.simSegment = row.simSegment,
        rel.simConcept = row.simConcept,
        rel.simRefinedLabel = row.simRefinedLabel,

        rel.sharedStrength = row.sharedStrength,
        rel.sharedConcepts = row.sharedConcepts,
        rel.sharedConceptScores = row.sharedConceptScores,

        rel.srcLabel = row.srcLabel,
        rel.dstLabel = row.dstLabel,

        rel.srcOriginalLabel = row.srcOriginalLabel,
        rel.dstOriginalLabel = row.dstOriginalLabel,

        rel.srcRefinedLabel = row.srcRefinedLabel,
        rel.dstRefinedLabel = row.dstRefinedLabel,

        rel.srcDisplayLabel = row.srcDisplayLabel,
        rel.dstDisplayLabel = row.dstDisplayLabel,

        rel.srcSemanticLabel = row.srcSemanticLabel,
        rel.dstSemanticLabel = row.dstSemanticLabel
    """

    total = 0

    with driver.session(database="neo4j") as session:
        for batch in chunks(cleaned, SEGMENT_SIM_BATCH_SIZE):
            session.run(query, rows=batch).consume()
            total += len(batch)

    print(f"✅ Imported segment similarities: {total} (skipped {skipped})")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    create_constraints()

    import_images(r"Scenes\output_scenesEmb\images_out.json")
    import_image_locations(r"Scenes\cams_descriptions_fixed_finland.json")
    import_image_edges(r"Scenes\output_scenesEmb\edges_out.json")

    import_segments(r"Segmentation\output_segmentsEmb\segments_out.json")
    import_segment_similarity(r"Segmentation\output_segmentsEmb\segment_edges_out.json")

    driver.close()