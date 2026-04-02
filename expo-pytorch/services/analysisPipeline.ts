import * as ImageManipulator from "expo-image-manipulator";
import * as MediaLibrary from "expo-media-library";
import pako from "pako";
import { ScalarType } from "react-native-executorch";
import type { TensorPtr } from "react-native-executorch";

import type {
  AnalysisMode,
  AnalysisStage,
  BBox,
  ClassificationResult,
  FaceResult,
  ImageAnalysisResult,
} from "@/types/workflow";

const BLAZEFACE_INPUT_SIZE = 128;
const VIT_INPUT_SIZE = 224;
const VIT_MEAN = [0.485, 0.456, 0.406];
const VIT_STD = [0.229, 0.224, 0.225];
const AGE_LABELS = [
  "0-2",
  "3-9",
  "10-19",
  "20-29",
  "30-39",
  "40-49",
  "50-59",
  "60-69",
  "more than 70",
];
const GENDER_LABELS = ["female", "male"];
const BLAZEFACE_NUM_ANCHORS = 896;
const SCORE_THRESHOLD = 0.75;
const NMS_IOU_THRESHOLD = 0.3;

export type ForwardRunner = (inputs: TensorPtr[]) => Promise<TensorPtr[]>;

export type AnalysisModules = {
  faceDetectorForward: ForwardRunner;
  ageModelForward: ForwardRunner;
  genderModelForward: ForwardRunner;
  nsfwModelForward: ForwardRunner;
  nsfwLabels: string[];
};

export type AnalysisProgressEvent = {
  assetId: string;
  uri: string;
  stage: AnalysisStage;
  progress: number;
  faceCount?: number;
  error?: string;
};

export type AnalyzeAssetOptions = {
  asset: MediaLibrary.Asset;
  mode: AnalysisMode;
  modules: AnalysisModules;
  onProgress?: (event: AnalysisProgressEvent) => void;
  shouldCancel?: () => boolean;
};

export class AnalysisCancelledError extends Error {
  constructor() {
    super("Analysis cancelled");
    this.name = "AnalysisCancelledError";
  }
}

function emitProgress(
  options: AnalyzeAssetOptions,
  stage: AnalysisStage,
  progress: number,
  extras: Partial<AnalysisProgressEvent> = {},
): void {
  options.onProgress?.({
    assetId: options.asset.id,
    uri: options.asset.uri,
    stage,
    progress,
    ...extras,
  });
}

function assertNotCancelled(shouldCancel?: () => boolean): void {
  if (shouldCancel?.()) {
    throw new AnalysisCancelledError();
  }
}

function generateAnchors(): number[][] {
  const anchors: number[][] = [];

  for (let y = 0; y < 16; y += 1) {
    for (let x = 0; x < 16; x += 1) {
      for (let k = 0; k < 2; k += 1) {
        anchors.push([(x + 0.5) / 16, (y + 0.5) / 16, 1, 1]);
      }
    }
  }

  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      for (let k = 0; k < 6; k += 1) {
        anchors.push([(x + 0.5) / 8, (y + 0.5) / 8, 1, 1]);
      }
    }
  }

  return anchors;
}

const ANCHORS = generateAnchors();

function readUint32BE(bytes: Uint8Array, offset: number): number {
  return (
    ((bytes[offset] << 24) |
      (bytes[offset + 1] << 16) |
      (bytes[offset + 2] << 8) |
      bytes[offset + 3]) >>>
    0
  );
}

function applyPNGFilter(
  filterType: number,
  raw: Uint8Array,
  prev: Uint8Array,
  out: Uint8Array,
  bpp: number,
): void {
  const size = raw.length;

  switch (filterType) {
    case 0:
      out.set(raw);
      break;
    case 1:
      for (let index = 0; index < size; index += 1) {
        out[index] = (raw[index] + (index >= bpp ? out[index - bpp] : 0)) & 0xff;
      }
      break;
    case 2:
      for (let index = 0; index < size; index += 1) {
        out[index] = (raw[index] + prev[index]) & 0xff;
      }
      break;
    case 3:
      for (let index = 0; index < size; index += 1) {
        const left = index >= bpp ? out[index - bpp] : 0;
        out[index] = (raw[index] + Math.floor((left + prev[index]) / 2)) & 0xff;
      }
      break;
    case 4:
      for (let index = 0; index < size; index += 1) {
        const left = index >= bpp ? out[index - bpp] : 0;
        const up = prev[index];
        const upLeft = index >= bpp ? prev[index - bpp] : 0;
        const predictor = left + up - upLeft;
        const leftDistance = Math.abs(predictor - left);
        const upDistance = Math.abs(predictor - up);
        const upLeftDistance = Math.abs(predictor - upLeft);
        const paeth =
          leftDistance <= upDistance && leftDistance <= upLeftDistance
            ? left
            : upDistance <= upLeftDistance
              ? up
              : upLeft;
        out[index] = (raw[index] + paeth) & 0xff;
      }
      break;
    default:
      throw new Error(`Unsupported PNG filter type: ${filterType}`);
  }
}

function decodePNG(pngBytes: Uint8Array): {
  pixels: Uint8Array;
  width: number;
  height: number;
  channels: number;
} {
  let offset = 8;
  let width = 0;
  let height = 0;
  let channels = 3;
  const idatChunks: Uint8Array[] = [];

  while (offset < pngBytes.length) {
    const length = readUint32BE(pngBytes, offset);
    offset += 4;

    const type = String.fromCharCode(
      pngBytes[offset],
      pngBytes[offset + 1],
      pngBytes[offset + 2],
      pngBytes[offset + 3],
    );
    offset += 4;

    const data = pngBytes.slice(offset, offset + length);
    offset += length + 4;

    if (type === "IHDR") {
      width = readUint32BE(data, 0);
      height = readUint32BE(data, 4);
      channels = data[9] === 6 ? 4 : 3;
      continue;
    }

    if (type === "IDAT") {
      idatChunks.push(data);
      continue;
    }

    if (type === "IEND") {
      break;
    }
  }

  const totalLength = idatChunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const joinedIdat = new Uint8Array(totalLength);
  let position = 0;

  for (const chunk of idatChunks) {
    joinedIdat.set(chunk, position);
    position += chunk.length;
  }

  const raw = pako.inflate(joinedIdat) as Uint8Array;
  const rowBytes = width * channels;
  const pixels = new Uint8Array(width * height * channels);
  const emptyRow = new Uint8Array(rowBytes);

  for (let row = 0; row < height; row += 1) {
    const rowStart = row * (rowBytes + 1);
    const filterType = raw[rowStart];
    const rowData = raw.slice(rowStart + 1, rowStart + rowBytes + 1);
    const outRow = pixels.subarray(row * rowBytes, (row + 1) * rowBytes);
    const prevRow = row > 0 ? pixels.subarray((row - 1) * rowBytes, row * rowBytes) : emptyRow;

    applyPNGFilter(filterType, rowData, prevRow, outRow, channels);
  }

  return { pixels, width, height, channels };
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = globalThis.atob(base64);
  const bytes = new Uint8Array(binary.length);

  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return bytes;
}

async function imageUriToTensor(uri: string, size: number): Promise<Float32Array> {
  const resized = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width: size, height: size } }],
    { base64: true, format: ImageManipulator.SaveFormat.PNG },
  );

  if (!resized.base64) {
    throw new Error("imageUriToTensor: no base64 returned");
  }

  const { pixels, channels } = decodePNG(base64ToBytes(resized.base64));
  const numPixels = size * size;
  const tensor = new Float32Array(3 * numPixels);

  for (let index = 0; index < numPixels; index += 1) {
    tensor[index] = pixels[index * channels] / 127.5 - 1;
    tensor[numPixels + index] = pixels[index * channels + 1] / 127.5 - 1;
    tensor[2 * numPixels + index] = pixels[index * channels + 2] / 127.5 - 1;
  }

  return tensor;
}

async function imageUriToViTTensor(uri: string): Promise<Float32Array> {
  const resized = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width: VIT_INPUT_SIZE, height: VIT_INPUT_SIZE } }],
    { base64: true, format: ImageManipulator.SaveFormat.PNG },
  );

  if (!resized.base64) {
    throw new Error("imageUriToViTTensor: no base64 returned");
  }

  const { pixels, channels } = decodePNG(base64ToBytes(resized.base64));
  const numPixels = VIT_INPUT_SIZE * VIT_INPUT_SIZE;
  const tensor = new Float32Array(3 * numPixels);

  for (let index = 0; index < numPixels; index += 1) {
    tensor[index] = (pixels[index * channels] / 255 - VIT_MEAN[0]) / VIT_STD[0];
    tensor[numPixels + index] =
      (pixels[index * channels + 1] / 255 - VIT_MEAN[1]) / VIT_STD[1];
    tensor[2 * numPixels + index] =
      (pixels[index * channels + 2] / 255 - VIT_MEAN[2]) / VIT_STD[2];
  }

  return tensor;
}

function topFromLogits(logits: Float32Array, labels: string[]): ClassificationResult {
  const max = Math.max(...Array.from(logits));
  const exps = Array.from(logits).map((value) => Math.exp(value - max));
  const sum = exps.reduce((accumulator, value) => accumulator + value, 0);
  const probabilities = exps.map((value) => value / sum);

  let bestIndex = 0;

  for (let index = 1; index < probabilities.length; index += 1) {
    if (probabilities[index] > probabilities[bestIndex]) {
      bestIndex = index;
    }
  }

  return {
    label: labels[bestIndex] ?? `class_${bestIndex}`,
    score: probabilities[bestIndex],
  };
}

function allFromLogits(logits: Float32Array, labels: string[]): ClassificationResult[] {
  const max = Math.max(...Array.from(logits));
  const exps = Array.from(logits).map((value) => Math.exp(value - max));
  const sum = exps.reduce((accumulator, value) => accumulator + value, 0);
  const probabilities = exps.map((value) => value / sum);

  return probabilities
    .map((score, index) => ({
      label: labels[index] ?? `class_${index}`,
      score,
    }))
    .sort((left, right) => right.score - left.score);
}

function decodeBoxes(rawBoxes: Float32Array, anchors: number[][]): number[][] {
  const boxes: number[][] = [];

  for (let index = 0; index < BLAZEFACE_NUM_ANCHORS; index += 1) {
    const [anchorCx, anchorCy, anchorW, anchorH] = anchors[index];
    const xCenter = rawBoxes[index * 16] / BLAZEFACE_INPUT_SIZE * anchorW + anchorCx;
    const yCenter = rawBoxes[index * 16 + 1] / BLAZEFACE_INPUT_SIZE * anchorH + anchorCy;
    const width = rawBoxes[index * 16 + 2] / BLAZEFACE_INPUT_SIZE * anchorW;
    const height = rawBoxes[index * 16 + 3] / BLAZEFACE_INPUT_SIZE * anchorH;

    boxes.push([
      yCenter - height / 2,
      xCenter - width / 2,
      yCenter + height / 2,
      xCenter + width / 2,
    ]);
  }

  return boxes;
}

function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

function iou(left: number[], right: number[]): number {
  const interY1 = Math.max(left[0], right[0]);
  const interX1 = Math.max(left[1], right[1]);
  const interY2 = Math.min(left[2], right[2]);
  const interX2 = Math.min(left[3], right[3]);
  const intersection = Math.max(0, interY2 - interY1) * Math.max(0, interX2 - interX1);
  const leftArea = (left[2] - left[0]) * (left[3] - left[1]);
  const rightArea = (right[2] - right[0]) * (right[3] - right[1]);

  return intersection / (leftArea + rightArea - intersection + 1e-6);
}

function nms(boxes: number[][], scores: number[], iouThreshold: number): number[] {
  const order = scores
    .map((score, index) => ({ score, index }))
    .sort((left, right) => right.score - left.score)
    .map((entry) => entry.index);

  const suppressed = new Set<number>();
  const keep: number[] = [];

  for (const candidateIndex of order) {
    if (suppressed.has(candidateIndex)) {
      continue;
    }

    keep.push(candidateIndex);

    for (const compareIndex of order) {
      if (compareIndex === candidateIndex || suppressed.has(compareIndex)) {
        continue;
      }

      if (iou(boxes[candidateIndex], boxes[compareIndex]) > iouThreshold) {
        suppressed.add(compareIndex);
      }
    }
  }

  return keep;
}

function postprocessBlazeFace(outputs: TensorPtr[]): BBox[] {
  if (outputs.length < 2) {
    return [];
  }

  const rawBoxes = new Float32Array(outputs[0].dataPtr as ArrayBuffer);
  const rawScores = new Float32Array(outputs[1].dataPtr as ArrayBuffer);
  const candidateIndices: number[] = [];

  for (let index = 0; index < BLAZEFACE_NUM_ANCHORS; index += 1) {
    if (sigmoid(rawScores[index]) >= SCORE_THRESHOLD) {
      candidateIndices.push(index);
    }
  }

  if (candidateIndices.length === 0) {
    return [];
  }

  const decodedBoxes = decodeBoxes(rawBoxes, ANCHORS);
  const candidateBoxes = candidateIndices.map((index) => decodedBoxes[index]);
  const candidateScores = candidateIndices.map((index) => sigmoid(rawScores[index]));
  const keptIndices = nms(candidateBoxes, candidateScores, NMS_IOU_THRESHOLD);

  return keptIndices.map((keepIndex) => {
    const [ymin, xmin, ymax, xmax] = candidateBoxes[keepIndex];

    return {
      ymin: Math.max(0, ymin),
      xmin: Math.max(0, xmin),
      ymax: Math.min(1, ymax),
      xmax: Math.min(1, xmax),
    };
  });
}

async function cropFace(
  photoUri: string,
  bbox: BBox,
  photoWidth: number,
  photoHeight: number,
): Promise<string> {
  const padX = (bbox.xmax - bbox.xmin) * 0.2;
  const padY = (bbox.ymax - bbox.ymin) * 0.2;
  const x1 = Math.max(0, (bbox.xmin - padX) * photoWidth);
  const y1 = Math.max(0, (bbox.ymin - padY) * photoHeight);
  const x2 = Math.min(photoWidth, (bbox.xmax + padX) * photoWidth);
  const y2 = Math.min(photoHeight, (bbox.ymax + padY) * photoHeight);

  const result = await ImageManipulator.manipulateAsync(
    photoUri,
    [{ crop: { originX: x1, originY: y1, width: x2 - x1, height: y2 - y1 } }],
    { compress: 0.9, format: ImageManipulator.SaveFormat.JPEG },
  );

  return result.uri;
}

export async function analyzeAsset(options: AnalyzeAssetOptions): Promise<ImageAnalysisResult> {
  const {
    asset,
    modules: { ageModelForward, faceDetectorForward, genderModelForward, nsfwLabels, nsfwModelForward },
  } = options;

  assertNotCancelled(options.shouldCancel);
  emitProgress(options, "running_nsfw", 0.14);

  const nsfwInput = await imageUriToViTTensor(asset.uri);
  const nsfwOutputs = await nsfwModelForward([
    {
      dataPtr: nsfwInput,
      scalarType: ScalarType.FLOAT,
      sizes: [1, 3, VIT_INPUT_SIZE, VIT_INPUT_SIZE],
    },
  ]);

  const nsfwLogits = new Float32Array(nsfwOutputs[0].dataPtr as ArrayBuffer);
  const nsfw = allFromLogits(nsfwLogits, nsfwLabels);

  assertNotCancelled(options.shouldCancel);
  emitProgress(options, "running_face_detection", 0.34);

  const faceDetectionInput = await imageUriToTensor(asset.uri, BLAZEFACE_INPUT_SIZE);
  const detectorOutputs = await faceDetectorForward([
    {
      dataPtr: faceDetectionInput,
      scalarType: ScalarType.FLOAT,
      sizes: [1, 3, BLAZEFACE_INPUT_SIZE, BLAZEFACE_INPUT_SIZE],
    },
  ]);

  const bboxes = postprocessBlazeFace(detectorOutputs);

  if (bboxes.length === 0) {
    emitProgress(options, "completed", 1, { faceCount: 0 });

    return {
      analyzedAt: Date.now(),
      assetId: asset.id,
      faces: [],
      nsfw,
      uri: asset.uri,
    };
  }

  emitProgress(options, "cropping_faces", 0.52, { faceCount: bboxes.length });
  const faces: FaceResult[] = [];

  for (let index = 0; index < bboxes.length; index += 1) {
    assertNotCancelled(options.shouldCancel);

    const bbox = bboxes[index];
    const croppedUri = await cropFace(asset.uri, bbox, asset.width, asset.height);

    emitProgress(options, "running_demographics", 0.64 + (index / bboxes.length) * 0.26, {
      faceCount: bboxes.length,
    });

    const vitTensor = await imageUriToViTTensor(croppedUri);
    const vitTensorPtr: TensorPtr = {
      dataPtr: vitTensor,
      scalarType: ScalarType.FLOAT,
      sizes: [1, 3, VIT_INPUT_SIZE, VIT_INPUT_SIZE],
    };

    const [ageOutputs, genderOutputs] = await Promise.all([
      ageModelForward([vitTensorPtr]),
      genderModelForward([vitTensorPtr]),
    ]);

    const ageLogits = new Float32Array(ageOutputs[0].dataPtr as ArrayBuffer);
    const genderLogits = new Float32Array(genderOutputs[0].dataPtr as ArrayBuffer);

    faces.push({
      age: topFromLogits(ageLogits, AGE_LABELS),
      bbox,
      gender: topFromLogits(genderLogits, GENDER_LABELS),
    });
  }

  emitProgress(options, "completed", 1, { faceCount: faces.length });

  return {
    analyzedAt: Date.now(),
    assetId: asset.id,
    faces,
    nsfw,
    uri: asset.uri,
  };
}

export function estimateAnalysisDuration(assetCount: number, mode: AnalysisMode): number {
  if (assetCount === 0) {
    return 0;
  }

  const perAssetSeconds = mode === "quick" ? 7 : 9;
  return assetCount * perAssetSeconds;
}

export function formatClassificationResults(results: ClassificationResult[], limit = 2): string {
  return results
    .slice(0, limit)
    .map((result) => `${result.label} ${(result.score * 100).toFixed(0)}%`)
    .join(" • ");
}
