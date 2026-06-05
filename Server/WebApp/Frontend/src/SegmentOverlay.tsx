import { useEffect, useRef, useState } from "react";
import type { SegmentInfo, SegmentOverlayProps } from "./InfoTypes";
import { API_BASE } from "./config";

function toBackendUrl(url: string | null | undefined): string {
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.startsWith("/")) return `${API_BASE}${url}`;
  return `${API_BASE}/${url}`;
}

export default function SegmentOverlay({
  image,
  segments,
  hoveredSegmentUid,
  selectedSegmentUid,
  focusSegmentUid = null,
  isFocusMode = false,
  revealProgress = 1,
  focusTransitionMs = 900,
  hoverLabelMode = "label",
  dbVariant = "main",
  onFocusReady,
  onSegmentClick,
  onSegmentHover,
  onImageClick,
}: SegmentOverlayProps) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const focusCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // The segmap is loaded once per image and stored as segment IDs for fast hover/click lookup.
  const segmapArrayRef = useRef<Uint32Array | null>(null);
  const segmapSizeRef = useRef<{ width: number; height: number } | null>(null);
  const lastHoveredRef = useRef<string | null>(null);

  const [segmapReady, setSegmapReady] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string } | null>(null);

  const activeFocusSegmentUid = focusSegmentUid ?? selectedSegmentUid;

  function resolveFocusSegment(): SegmentInfo | undefined {
    if (activeFocusSegmentUid) {
      const byUid = segments.find((segment) => segment.segmentUid === activeFocusSegmentUid);
      if (byUid) return byUid;
    }

    if (image.matchedSegment?.segmentId != null) {
      const bySegmentId = segments.find(
        (segment) => segment.segmentId === image.matchedSegment?.segmentId,
      );
      if (bySegmentId) return bySegmentId;
    }

    return image.matchedSegment;
  }

  const focusSegment = resolveFocusSegment();

  useEffect(() => {
    setSegmapReady(false);
    setImageLoaded(false);
    setTooltip(null);
    segmapArrayRef.current = null;
    segmapSizeRef.current = null;
    lastHoveredRef.current = null;

    const segmapImg = new Image();
    segmapImg.crossOrigin = "anonymous";

    segmapImg.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = segmapImg.width;
      canvas.height = segmapImg.height;

      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) return;

      ctx.drawImage(segmapImg, 0, 0);

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      const arr = new Uint32Array(canvas.width * canvas.height);

      // Segment IDs are encoded in the RGB channels of the segmap PNG.
      for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
        arr[p] = data[i] + data[i + 1] * 256 + data[i + 2] * 65536;
      }

      segmapArrayRef.current = arr;
      segmapSizeRef.current = { width: canvas.width, height: canvas.height };
      setSegmapReady(true);
    };

    segmapImg.onerror = () => {
      console.error(`Could not load segmap for image ${image.imageUid} from db ${dbVariant}`);
    };

    segmapImg.src = `${API_BASE}/api/segments/segmap-png?imageUid=${encodeURIComponent(
      image.imageUid,
    )}&db=${encodeURIComponent(dbVariant)}`;
  }, [image.imageUid, dbVariant]);

  function getOriginalCoords(e: React.MouseEvent<HTMLDivElement>) {
    const img = imgRef.current;
    const segmapSize = segmapSizeRef.current;
    const targetWidth = segmapSize?.width ?? image.width;
    const targetHeight = segmapSize?.height ?? image.height;

    if (!img || !targetWidth || !targetHeight) return null;

    const rect = img.getBoundingClientRect();
    const xInImg = e.clientX - rect.left;
    const yInImg = e.clientY - rect.top;

    if (xInImg < 0 || yInImg < 0 || xInImg >= rect.width || yInImg >= rect.height) {
      return null;
    }

    return {
      x: Math.floor((xInImg / rect.width) * targetWidth),
      y: Math.floor((yInImg / rect.height) * targetHeight),
    };
  }

  function getHoverLabel(segment: SegmentInfo): string {
    if (hoverLabelMode === "refined") {
      return segment.refinedLabel?.trim() || segment.label?.trim() || "Unknown segment";
    }

    return segment.label?.trim() || "Unknown segment";
  }

  function getSegmentIdAtPoint(x: number, y: number): number | null {
    const arr = segmapArrayRef.current;
    const size = segmapSizeRef.current;

    if (!arr || !size) return null;
    if (x < 0 || y < 0 || x >= size.width || y >= size.height) return null;

    const id = arr[y * size.width + x];
    return id > 0 ? id : null;
  }

  function clearFocusCanvas() {
    const canvas = focusCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function drawFocusCanvas(segment: SegmentInfo | undefined): number {
    const canvas = focusCanvasRef.current;
    const img = imgRef.current;
    const arr = segmapArrayRef.current;
    const size = segmapSizeRef.current;

    if (!canvas || !img || !arr || !size) return 0;

    const rect = img.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width));
    canvas.height = Math.max(1, Math.round(rect.height));

    const ctx = canvas.getContext("2d");
    if (!ctx) return 0;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!isFocusMode) return 0;

    if (!segment || segment.segmentId == null) {
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      return 0;
    }

    const targetId = segment.segmentId;
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;
    let paintedPixels = 0;

    for (let cy = 0; cy < canvas.height; cy += 1) {
      const sy = Math.min(size.height - 1, Math.floor((cy / canvas.height) * size.height));

      for (let cx = 0; cx < canvas.width; cx += 1) {
        const sx = Math.min(size.width - 1, Math.floor((cx / canvas.width) * size.width));
        const segId = arr[sy * size.width + sx];
        const out = (cy * canvas.width + cx) * 4;

        data[out] = 0;
        data[out + 1] = 0;
        data[out + 2] = 0;

        if (segId === targetId) {
          data[out + 3] = 0;
          paintedPixels += 1;
        } else {
          data[out + 3] = 255;
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
    return paintedPixels;
  }

  function notifyFocusReady(paintedPixels: number) {
    if (paintedPixels <= 0) {
      console.warn("Focus segment could not be drawn", {
        imageUid: image.imageUid,
        dbVariant,
        activeFocusSegmentUid,
        matchedSegment: image.matchedSegment,
        segmentCount: segments.length,
      });
      return;
    }

    onFocusReady?.({
      imageUid: image.imageUid,
      focusSegmentUid: focusSegment?.segmentUid ?? activeFocusSegmentUid,
      segmentId: focusSegment?.segmentId ?? null,
      paintedPixels,
    });
  }

  useEffect(() => {
    if (!isFocusMode) {
      clearFocusCanvas();
      return;
    }

    if (!segmapReady || !imageLoaded) return;
    notifyFocusReady(drawFocusCanvas(focusSegment));
  }, [
    isFocusMode,
    activeFocusSegmentUid,
    segmapReady,
    imageLoaded,
    image.imageUid,
    image.width,
    image.height,
    segments,
    image.matchedSegment,
    dbVariant,
  ]);

  useEffect(() => {
    const handleResize = () => {
      if (isFocusMode) drawFocusCanvas(focusSegment);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [isFocusMode, activeFocusSegmentUid, segmapReady, imageLoaded]);

  useEffect(() => {
    const img = imgRef.current;
    if (img?.complete && img.naturalWidth > 0) {
      setImageLoaded(true);
    }
  }, [image.imageUid]);

  function handleMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    if (!segmapReady || isFocusMode) return;

    const coords = getOriginalCoords(e);

    if (!coords) {
      if (lastHoveredRef.current !== null) {
        lastHoveredRef.current = null;
        onSegmentHover(null);
      }
      setTooltip(null);
      return;
    }

    const segmentId = getSegmentIdAtPoint(coords.x, coords.y);
    const segment = segments.find((s) => s.segmentId === segmentId);
    const nextUid = segment?.segmentUid ?? null;

    if (segment) {
      const rootRect = e.currentTarget.getBoundingClientRect();
      setTooltip({
        x: e.clientX - rootRect.left + 12,
        y: e.clientY - rootRect.top + 12,
        label: getHoverLabel(segment),
      });
    } else {
      setTooltip(null);
    }

    if (nextUid !== lastHoveredRef.current) {
      lastHoveredRef.current = nextUid;
      onSegmentHover(nextUid);
    }
  }

  function handleClick(e: React.MouseEvent<HTMLDivElement>) {
    if (!segmapReady || isFocusMode) return;

    const coords = getOriginalCoords(e);
    if (!coords) return;

    const segmentId = getSegmentIdAtPoint(coords.x, coords.y);
    const segment = segments.find((s) => s.segmentId === segmentId);

    if (segment) {
      onSegmentClick(segment);
    } else {
      onImageClick?.();
    }
  }

  function handleContextMenu(e: React.MouseEvent<HTMLDivElement>) {
    e.preventDefault();
    if (isFocusMode) return;

    setTooltip(null);
    lastHoveredRef.current = null;
    onSegmentHover(null);
    onImageClick?.();
  }

  function handleMouseLeave() {
    lastHoveredRef.current = null;
    setTooltip(null);
    onSegmentHover(null);
  }

  return (
    <div
      className="fts-segment-overlay-root"
      onMouseMove={handleMouseMove}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
      onMouseLeave={handleMouseLeave}
    >
      <img
        ref={imgRef}
        src={toBackendUrl(image.imageUrl)}
        alt={image.imageUid}
        className="fts-segment-base-image"
        draggable={false}
        onLoad={() => {
          setImageLoaded(true);
          if (isFocusMode) {
            window.requestAnimationFrame(() => notifyFocusReady(drawFocusCanvas(focusSegment)));
          }
        }}
      />

      <canvas
        ref={focusCanvasRef}
        className="fts-segment-focus-canvas"
        style={{
          opacity: isFocusMode ? 1 - revealProgress : 0,
          transitionDuration: `${focusTransitionMs}ms`,
        }}
      />

      {!isFocusMode && tooltip && hoveredSegmentUid && (
        <div className="fts-segment-hover-label" style={{ left: tooltip.x, top: tooltip.y }}>
          {tooltip.label}
        </div>
      )}

      {!segmapReady && <div className="fts-segment-loading-badge">Loading segments...</div>}
    </div>
  );
}
