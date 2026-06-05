import { useEffect, useRef, useState } from "react";
import { MapContainer, Marker, TileLayer, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import markerIcon2x from "leaflet/dist/images/marker-icon-2x.png";
import markerIcon from "leaflet/dist/images/marker-icon.png";
import markerShadow from "leaflet/dist/images/marker-shadow.png";

import SegmentOverlay from "./SegmentOverlay";
import type { DbVariant, FocusReadyInfo, ImageInfo, MapPin, SegmentInfo } from "./InfoTypes";
import { API_BASE } from "./config";
import "./gameMap.css";

// Leaflet needs the marker images to be configured manually when used with Vite/Webpack.
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

const currentPinIcon = new L.Icon({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
  className: "fts-current-leaflet-pin",
  iconSize: [32, 52],
  iconAnchor: [16, 52],
  popupAnchor: [1, -42],
  shadowSize: [41, 41],
});

const oldPinIcon = new L.Icon({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
  className: "fts-old-leaflet-pin",
  iconSize: [24, 39],
  iconAnchor: [12, 39],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

const WORLD_CENTER: [number, number] = [-10, 0];
const WORLD_ZOOM = 0.8;
const WORLD_BOUNDS: [[number, number], [number, number]] = [
  [-85, -180],
  [85, 180],
];

const INCLUDE_SAME_CAMERA = false;
const MAX_PIN_GROUP_DISTANCE_METERS = 50;
const FIRST_FOCUS_ANIMATION_MS = 1000;
const SECOND_REVEAL_ANIMATION_MS = 2000;
const OVERLAY_READY_TIMEOUT_MS = 1500;

const DEFAULT_DESCRIPTION =
  "Explore webcam images through visual similarity. Select a segment in the image or right-click to select the whole scene. To reset the map and get a new random image, click the map.";

type GameMapProps = {
  dbVariant?: DbVariant;
  title?: string;
  description?: string;
};

function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function waitForNextFrame() {
  return new Promise((resolve) => window.requestAnimationFrame(() => resolve(null)));
}

function preloadImage(src: string) {
  return new Promise<void>((resolve) => {
    const img = new Image();
    img.onload = () => resolve();
    img.onerror = () => resolve();
    img.src = src;
  });
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function toBackendUrl(url: string | null | undefined): string {
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/")) return `${API_BASE}${url}`;
  return `${API_BASE}/${url}`;
}

function getPosition(image: ImageInfo | null): [number, number] | null {
  if (!image || typeof image.latitude !== "number" || typeof image.longitude !== "number") {
    return null;
  }

  return [image.latitude, image.longitude];
}

function getCameraIdFromImageUid(imageUid: string): string {
  return imageUid.split("_")[0];
}

function buildExcludeCameraParams(pins: MapPin[]) {
  const cameraIds = Array.from(new Set(pins.map((pin) => getCameraIdFromImageUid(pin.imageUid))));

  return cameraIds
    .map((cameraId) => `excludeCameras=${encodeURIComponent(cameraId)}`)
    .join("&");
}

function resolveFocusSegmentUid(nextImage: ImageInfo): string | null {
  const matchedSegment = nextImage.matchedSegment;
  if (!matchedSegment) return null;
  if (matchedSegment.segmentUid) return matchedSegment.segmentUid;

  const sameSegment = nextImage.segments?.find(
    (segment) => segment.segmentId === matchedSegment.segmentId,
  );

  return sameSegment?.segmentUid ?? null;
}

function distanceMeters(a: [number, number], b: [number, number]): number {
  const earthRadius = 6371000;
  const lat1 = (a[0] * Math.PI) / 180;
  const lat2 = (b[0] * Math.PI) / 180;
  const deltaLat = ((b[0] - a[0]) * Math.PI) / 180;
  const deltaLng = ((b[1] - a[1]) * Math.PI) / 180;

  const sinLat = Math.sin(deltaLat / 2);
  const sinLng = Math.sin(deltaLng / 2);
  const h = sinLat * sinLat + Math.cos(lat1) * Math.cos(lat2) * sinLng * sinLng;

  return earthRadius * 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
}

function groupPinsByPosition(pins: MapPin[]) {
  const groups: { key: string; position: [number, number]; pins: MapPin[] }[] = [];

  for (const pin of pins) {
    const existingGroup = groups.find(
      (group) => distanceMeters(group.position, pin.position) <= MAX_PIN_GROUP_DISTANCE_METERS,
    );

    if (existingGroup) {
      existingGroup.pins.push(pin);
    } else {
      groups.push({
        key: `${pin.position[0]},${pin.position[1]}`,
        position: pin.position,
        pins: [pin],
      });
    }
  }

  return groups;
}

function MapClickHandler({ onClick }: { onClick: (lat: number, lng: number) => void }) {
  useMapEvents({
    click: (event) => onClick(event.latlng.lat, event.latlng.lng),
  });

  return null;
}

function MapSizeFix() {
  const map = useMap();

  useEffect(() => {
    const updateSize = () => {
      map.invalidateSize();
      map.fitBounds(WORLD_BOUNDS, { padding: [6, 6], animate: false });

      if (map.getZoom() < WORLD_ZOOM) {
        map.setZoom(WORLD_ZOOM, { animate: false });
      }
    };

    const timeoutId = window.setTimeout(updateSize, 150);
    window.addEventListener("resize", updateSize);

    return () => {
      window.clearTimeout(timeoutId);
      window.removeEventListener("resize", updateSize);
    };
  }, [map]);

  return null;
}

export default function GameMap({
  dbVariant = "main",
  title = "Travelling the World",
  description = DEFAULT_DESCRIPTION,
}: GameMapProps) {
  const [image, setImage] = useState<ImageInfo | null>(null);
  const [pins, setPins] = useState<MapPin[]>([]);
  const [selectedSegmentUid, setSelectedSegmentUid] = useState<string | null>(null);
  const [hoveredSegmentUid, setHoveredSegmentUid] = useState<string | null>(null);

  // These states control the black focus/reveal animation between two related segments.
  const [isFocusMode, setIsFocusMode] = useState(false);
  const [focusSegmentUid, setFocusSegmentUid] = useState<string | null>(null);
  const [revealProgress, setRevealProgress] = useState(1);
  const [transitioning, setTransitioning] = useState(false);
  const [imageSwitchBlackout, setImageSwitchBlackout] = useState(false);
  const [focusTransitionMs, setFocusTransitionMs] = useState(FIRST_FOCUS_ANIMATION_MS);

  const overlayReadyRef = useRef<FocusReadyInfo | null>(null);
  const groupedPins = groupPinsByPosition(pins);

  function resetFocusTransition() {
    setIsFocusMode(false);
    setFocusSegmentUid(null);
    setRevealProgress(1);
    setTransitioning(false);
    setImageSwitchBlackout(false);
    setFocusTransitionMs(FIRST_FOCUS_ANIMATION_MS);
  }

  function appendPinForImage(nextImage: ImageInfo) {
    const position = getPosition(nextImage);
    if (!position) return;

    setPins((prev) => {
      const alreadyExists = prev.some((pin) => pin.imageUid === nextImage.imageUid);
      return alreadyExists ? prev : [...prev, { imageUid: nextImage.imageUid, position }];
    });
  }

  function applyImage(
    nextImage: ImageInfo | null,
    resetPins = false,
    highlightSegmentUid: string | null = null,
  ) {
    if (!nextImage) return;

    const position = getPosition(nextImage);
    setImage(nextImage);
    setSelectedSegmentUid(highlightSegmentUid);
    setHoveredSegmentUid(null);

    if (resetPins) {
      setPins(position ? [{ imageUid: nextImage.imageUid, position }] : []);
      return;
    }

    appendPinForImage(nextImage);
  }

  async function loadRandomImage() {
    if (transitioning) return;

    setSelectedSegmentUid(null);
    setHoveredSegmentUid(null);
    resetFocusTransition();

    try {
      const data = await fetchJson<{ images?: ImageInfo[] }>(
        `${API_BASE}/api/images/random?n=1&db=${encodeURIComponent(dbVariant)}`,
      );
      const nextImage = data.images?.[0] ?? null;
      if (!nextImage) throw new Error("No image returned.");
      applyImage(nextImage, true);
    } catch (err) {
      console.error(err);
    }
  }

  async function loadSimilarScene() {
    if (!image || transitioning) return;

    setSelectedSegmentUid(null);
    setHoveredSegmentUid(null);
    resetFocusTransition();

    try {
      const excludeCameraParams = buildExcludeCameraParams(pins);
      const excludeSuffix = excludeCameraParams ? `&${excludeCameraParams}` : "";
      const data = await fetchJson<{ images?: ImageInfo[] }>(
        `${API_BASE}/api/images/similar?imageUid=${encodeURIComponent(image.imageUid)}` +
          `&n=1&includeSameCamera=${INCLUDE_SAME_CAMERA}` +
          `&db=${encodeURIComponent(dbVariant)}${excludeSuffix}`,
      );
      const nextImage = data.images?.[0] ?? null;
      if (!nextImage) throw new Error("No similar image returned.");
      applyImage(nextImage, false);
    } catch (err) {
      console.error(err);
    }
  }

  async function fetchSimilarSegmentImage(segment: SegmentInfo) {
    const excludeCameraParams = buildExcludeCameraParams(pins);
    const excludeCameraSuffix = excludeCameraParams ? `&${excludeCameraParams}` : "";

    const data = await fetchJson<{ images?: ImageInfo[] }>(
      `${API_BASE}/api/segments/similar?segmentUid=${encodeURIComponent(segment.segmentUid)}` +
        `&n=1&includeSameCamera=${INCLUDE_SAME_CAMERA}` +
        `&db=${encodeURIComponent(dbVariant)}${excludeCameraSuffix}`,
    );

    const nextImage = data.images?.[0] ?? null;
    if (!nextImage) throw new Error("No similar segment image returned.");
    return nextImage;
  }

  async function waitForOverlayReady(
    targetImageUid: string,
    targetFocusSegmentUid: string | null,
    targetSegmentId: number | null,
  ) {
    const startedAt = Date.now();

    while (Date.now() - startedAt < OVERLAY_READY_TIMEOUT_MS) {
      const ready = overlayReadyRef.current;

      if (ready?.imageUid === targetImageUid && ready.paintedPixels > 0) {
        const uidMatches = targetFocusSegmentUid !== null && ready.focusSegmentUid === targetFocusSegmentUid;
        const idMatches = targetSegmentId !== null && ready.segmentId === targetSegmentId;
        const noSpecificTarget = targetFocusSegmentUid === null && targetSegmentId === null;

        if (uidMatches || idMatches || noSpecificTarget) return true;
      }

      await waitForNextFrame();
    }

    console.warn("Overlay was not ready before timeout", {
      targetImageUid,
      targetFocusSegmentUid,
      targetSegmentId,
      ready: overlayReadyRef.current,
    });

    return false;
  }

  async function loadSimilarSegment(segment: SegmentInfo) {
    if (!image || transitioning) return;

    const currentImage = image;
    setTransitioning(true);
    setHoveredSegmentUid(null);
    setSelectedSegmentUid(segment.segmentUid);
    setImageSwitchBlackout(false);
    overlayReadyRef.current = null;

    try {
      const nextImage = await fetchSimilarSegmentImage(segment);
      const nextFocusSegmentUid = resolveFocusSegmentUid(nextImage);
      const nextFocusSegmentId = nextImage.matchedSegment?.segmentId ?? null;

      await preloadImage(toBackendUrl(nextImage.imageUrl));

      // First step: hide everything except the selected segment in the current image.
      setFocusTransitionMs(FIRST_FOCUS_ANIMATION_MS);
      setFocusSegmentUid(segment.segmentUid);
      setIsFocusMode(true);
      setRevealProgress(1);
      await waitForNextFrame();
      await waitForNextFrame();
      setRevealProgress(0);
      await wait(FIRST_FOCUS_ANIMATION_MS);

      // Second step: switch image while the screen is black, then reveal the matched segment.
      setImageSwitchBlackout(true);
      await waitForNextFrame();
      overlayReadyRef.current = null;
      applyImage(nextImage, false, nextFocusSegmentUid);

      setFocusTransitionMs(SECOND_REVEAL_ANIMATION_MS);
      setFocusSegmentUid(nextFocusSegmentUid);
      setIsFocusMode(true);
      setRevealProgress(0);

      const overlayReady = await waitForOverlayReady(nextImage.imageUid, nextFocusSegmentUid, nextFocusSegmentId);
      if (!overlayReady) {
        console.warn("Second focus overlay was not ready; revealing full image anyway.", {
          imageUid: nextImage.imageUid,
          nextFocusSegmentUid,
          nextFocusSegmentId,
          matchedSegment: nextImage.matchedSegment,
        });
      }

      await waitForNextFrame();
      await wait(250);
      setImageSwitchBlackout(false);
      await waitForNextFrame();
      setRevealProgress(1);
      await wait(SECOND_REVEAL_ANIMATION_MS);

      setIsFocusMode(false);
      setFocusSegmentUid(null);
      setRevealProgress(1);
      setFocusTransitionMs(FIRST_FOCUS_ANIMATION_MS);
      setTransitioning(false);
    } catch (err) {
      console.error(err);
      setImage(currentImage);
      resetFocusTransition();
    }
  }

  async function loadNearestImageFromMapClick(lat: number, lng: number) {
    if (transitioning) return;

    setSelectedSegmentUid(null);
    setHoveredSegmentUid(null);
    resetFocusTransition();

    try {
      const data = await fetchJson<{ image?: ImageInfo }>(
        `${API_BASE}/api/images/nearest?lat=${encodeURIComponent(lat)}` +
          `&lng=${encodeURIComponent(lng)}&db=${encodeURIComponent(dbVariant)}`,
      );
      const nextImage = data.image ?? null;
      if (!nextImage) throw new Error("No nearest image returned.");
      applyImage(nextImage, true);
    } catch (err) {
      console.error(err);
    }
  }

  useEffect(() => {
    void loadRandomImage();
  }, [dbVariant]);

  return (
    <div className="fts-map-page">
      <div className="fts-map-shell">
        <header className="fts-page-header">
          <h1>{title}</h1>
          <p>{description}</p>
        </header>

        <main className="fts-map-game-panel">
          <section className="fts-map-left-panel">
            {image && (
              <div className="fts-map-image-wrap">
                <div className="fts-map-image-stage">
                  <div className="fts-image-actual-frame">
                    <SegmentOverlay
                      key={`${dbVariant}-${image.imageUid}`}
                      image={image}
                      segments={image.segments ?? []}
                      dbVariant={dbVariant}
                      selectedSegmentUid={selectedSegmentUid}
                      hoveredSegmentUid={hoveredSegmentUid}
                      focusSegmentUid={focusSegmentUid}
                      isFocusMode={isFocusMode}
                      revealProgress={revealProgress}
                      focusTransitionMs={focusTransitionMs}
                      hoverLabelMode="refined"
                      onFocusReady={(ready) => {
                        overlayReadyRef.current = ready;
                      }}
                      onSegmentClick={(segment) => void loadSimilarSegment(segment)}
                      onImageClick={() => void loadSimilarScene()}
                      onSegmentHover={setHoveredSegmentUid}
                    />

                    {imageSwitchBlackout && <div className="fts-image-switch-blackout" />}
                  </div>
                </div>
              </div>
            )}
          </section>

          <aside className="fts-map-right-panel" aria-label="World map">
            <div className="fts-map-frame">
              <MapContainer
                center={WORLD_CENTER}
                zoom={WORLD_ZOOM}
                minZoom={0}
                maxZoom={6}
                zoomSnap={0.25}
                zoomDelta={0.25}
                className="fts-leaflet-map"
                preferCanvas
                scrollWheelZoom={false}
                worldCopyJump={false}
              >
                <MapSizeFix />
                <MapClickHandler onClick={(lat, lng) => void loadNearestImageFromMapClick(lat, lng)} />

                <TileLayer
                  attribution="&copy; OpenStreetMap contributors"
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  updateWhenIdle
                  updateWhenZooming={false}
                  keepBuffer={4}
                />

                {groupedPins.map((group) => {
                  const isCurrentGroup =
                    image !== null && group.pins.some((pin) => pin.imageUid === image.imageUid);

                  return (
                    <Marker
                      key={`${group.key}-${group.pins.map((pin) => pin.imageUid).join("-")}-${
                        isCurrentGroup ? "current" : "old"
                      }`}
                      position={group.position}
                      icon={isCurrentGroup ? currentPinIcon : oldPinIcon}
                    />
                  );
                })}
              </MapContainer>
            </div>
          </aside>
        </main>
      </div>
    </div>
  );
}
