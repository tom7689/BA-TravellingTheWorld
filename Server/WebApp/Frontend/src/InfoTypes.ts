export type DbVariant = "main" | "efficient" | "efficient_refined";

export type SegmentInfo = {
  segmentUid: string;
  segmentId: number;
  label?: string;
  originalLabel?: string;
  refinedLabel?: string;
  displayLabel?: string;
};

export type ImageInfo = {
  imageUid: string;
  imageUrl: string;
  width?: number;
  height?: number;

  webcamId?: string;
  latitude?: number;
  longitude?: number;
  city?: string;
  region?: string;
  countryName?: string;

  segments?: SegmentInfo[];
  matchedSegment?: SegmentInfo;
};

export type MapPin = {
  imageUid: string;
  position: [number, number];
};

export type FocusReadyInfo = {
  imageUid: string;
  focusSegmentUid: string | null;
  segmentId: number | null;
  paintedPixels: number;
};

export type SegmentOverlayProps = {
  image: ImageInfo;
  segments: SegmentInfo[];
  dbVariant?: DbVariant;

  hoveredSegmentUid: string | null;
  selectedSegmentUid: string | null;

  focusSegmentUid?: string | null;
  isFocusMode?: boolean;
  revealProgress?: number;
  focusTransitionMs?: number;
  hoverLabelMode?: "label" | "refined";

  onFocusReady?: (ready: FocusReadyInfo) => void;
  onSegmentClick: (segment: SegmentInfo) => void;
  onSegmentHover: (segmentUid: string | null) => void;
  onImageClick?: () => void;
};
