import * as MediaLibrary from "expo-media-library";

export type AnalysisMode = "quick" | "full";

export type BBox = {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
};

export type ClassificationResult = {
  label: string;
  score: number;
};

export type FaceResult = {
  bbox: BBox;
  age: ClassificationResult;
  gender: ClassificationResult;
};

export type ImageAnalysisResult = {
  assetId: string;
  uri: string;
  faces: FaceResult[];
  nsfw: ClassificationResult[];
  error?: string;
  analyzedAt: number;
};

export type PermissionState = {
  status: MediaLibrary.PermissionResponse["status"] | null;
  canAskAgain: boolean;
  accessPrivileges: MediaLibrary.PermissionResponse["accessPrivileges"] | null;
  isGranted: boolean;
  isLimited: boolean;
};

export type AnalysisStage =
  | "queued"
  | "running_nsfw"
  | "running_face_detection"
  | "cropping_faces"
  | "running_demographics"
  | "completed"
  | "failed"
  | "cancelled";

export type JobStatus =
  | "idle"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type AssetJobProgress = {
  assetId: string;
  uri: string;
  status: AnalysisStage;
  progress: number;
  stageLabel: string;
  faceCount?: number;
  error?: string;
  updatedAt: number;
};

export type AnalysisJobState = {
  status: JobStatus;
  activeAssetId: string | null;
  runId: number;
  total: number;
  completed: number;
  startedAt: number | null;
  completedAt: number | null;
  error: string | null;
};

export type AnalysisConfig = {
  mode: AnalysisMode;
};

export type ModelStatus = {
  key: "blazeface" | "age" | "gender" | "nsfw";
  label: string;
  isReady: boolean;
  downloadProgress: number;
  error: string | null;
};

export const ANALYSIS_STAGE_LABELS: Record<AnalysisStage, string> = {
  queued: "Queued",
  running_nsfw: "Running safety scan",
  running_face_detection: "Detecting faces",
  cropping_faces: "Preparing crops",
  running_demographics: "Estimating demographics",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

export const ANALYSIS_MODE_COPY: Record<
  AnalysisMode,
  {
    title: string;
    subtitle: string;
  }
> = {
  quick: {
    title: "Quick review",
    subtitle: "Current build still uses the full on-device stack while the UI stabilizes.",
  },
  full: {
    title: "Full review",
    subtitle: "Runs the current on-device NSFW, face, age, and gender pipeline.",
  },
};

export const INITIAL_JOB_STATE: AnalysisJobState = {
  status: "idle",
  activeAssetId: null,
  runId: 0,
  total: 0,
  completed: 0,
  startedAt: null,
  completedAt: null,
  error: null,
};
