
# BA-TravellingTheWorld

Dieses Repository enthält den Code zur Bachelorarbeit **Travelling The World**. Ziel des Projekts ist eine Webanwendung, mit der Bilder aus dem Datensatz anhand von Szenen- und Segmentähnlichkeiten explorativ verglichen werden können.

## Projektstruktur

Das Projekt ist in mehrere Ordner aufgeteilt, die nacheinander ausgeführt werden.

```text
Preparation/     Vorverarbeitung des Bilddatensatzes
Scenes/          Szenenerkennung und Bildähnlichkeiten
Segmentation/    Segmentierung, Segmentverfeinerung und Segmentähnlichkeiten
DB/              Import der erzeugten Daten in Neo4j
Server/          Docker-Anwendung mit Frontend, Backend und Neo4j
````

## Ablauf

### 1. Preparation

Im Ordner `Preparation` werden die Bilder für die weitere Verarbeitung vorbereitet.

Die wichtigsten Skripte sind:

* `ImageFilter.py`: entfernt zu dunkle oder ungeeignete Bilder
* `removeSmallImages.py`: entfernt Bilder mit zu geringer Auflösung
* `scenesToJson.py`: erstellt eine zentrale `scenes.json` mit den Bildpfaden und Metadaten

Die erzeugte `scenes.json` dient als Grundlage für die weiteren Schritte.

### 2. Scenes

Im Ordner `Scenes` werden die Szeneninformationen und Bildähnlichkeiten berechnet.

Das Skript `SceneEmb.py` führt die Szenenerkennung und die Ähnlichkeitsberechnung aus. Dabei werden unter anderem Places365 und OpenCLIP verwendet.

Als Ergebnis entstehen:

```text
images_out.json
image_edges_out.json
```

Diese Dateien enthalten die erkannten Szeneninformationen und die Ähnlichkeitsbeziehungen zwischen Bildern.

### 3. Segmentation

Im Ordner `Segmentation` werden die Bilder segmentiert und die Segmentähnlichkeiten berechnet.

Die Verarbeitung erfolgt in dieser Reihenfolge:

1. `Segmentation.py`
   Segmentiert die Bilder auf Basis der `scenes.json` und erstellt pro Bild eine JSON-Datei sowie eine Segmentkarte.

2. `SAM3correction.py`
   Verfeinert ausgewählte Segmente und erzeugt korrigierte Segmentdaten.

3. `Segrefinement.py`
   Erstellt verfeinerte Segment-Labels auf Basis der definierten Label-Kandidaten.

4. `segmentEmbedding.py`
   Berechnet die Ähnlichkeiten zwischen Segmenten.

Als Ergebnis entstehen unter anderem:

```text
segments_out.json
segment_edges_out.json
```

Diese Dateien werden später in die Neo4j-Datenbank importiert.

### 4. DB

Der Ordner `DB` enthält das Skript zum Import der erzeugten Daten in Neo4j.

Dabei werden die Ergebnisse aus `Scenes` und `Segmentation` zusammengeführt und als Knoten und Beziehungen in die Datenbank geschrieben.

Nach dem Import muss aus Neo4j ein `.dump`-File erstellt werden. Dieses Dump-File wird anschliessend für die Docker-Version der Anwendung verwendet.

### 5. Server

Der Ordner `Server` enthält die Webanwendung. Diese besteht aus:

* React/TypeScript-Frontend
* FastAPI-Backend
* Neo4j-Datenbank

Vor dem Start müssen folgende Daten vorhanden sein:

```text
ImagesAll2/                 Bilddatensatz
Segmentation/               JSON- und Segmentkarten-Dateien
neo4j/dump/                 Neo4j-Dump-Datei
```

Die Anwendung kann anschliessend mit Docker gestartet werden:

```bash
docker compose up -d --build
```

Das Frontend befindet sich unter:

```text
Webapp/Frontend
```

Die zentrale Anwendungslogik ist in `GameMap.tsx` umgesetzt. Weitere wichtige Dateien sind `SegmentOverlay.tsx`, `InfoTypes.ts` und das zugehörige CSS-File.

Das Backend befindet sich unter:

```text
Webapp/Backend
```

Die Datei `main.py` stellt die Schnittstelle zwischen Frontend und Neo4j-Datenbank bereit.

## Hinweise

Der vollständige Bilddatensatz sowie grosse erzeugte Dateien sind nicht im GitHub-Repository enthalten. Sie müssen lokal in die entsprechenden Ordner kopiert werden, bevor die Anwendung gestartet werden kann.

```
```
