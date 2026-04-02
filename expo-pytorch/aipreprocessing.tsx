import type { TensorPtr } from "react-native-executorch";
import * as ImageManipulator from "expo-image-manipulator";
import pako from "pako";

// ── BlazeFace constants ───────────────────────────────────────────────────────
const BLAZEFACE_INPUT_SIZE = 128;

// ── ViT constants (age + gender models) ──────────────────────────────────────
const VIT_INPUT_SIZE = 224;
// ImageNet normalization: mean and std per channel
const VIT_MEAN = [0.485, 0.456, 0.406];
const VIT_STD  = [0.229, 0.224, 0.225];
// Label sets from HuggingFace model configsc
const BLAZEFACE_NUM_ANCHORS = 896;
const SCORE_THRESHOLD = 0.75;
const NMS_IOU_THRESHOLD = 0.3;


type BBox = { ymin: number; xmin: number; ymax: number; xmax: number };

const ANCHORS = generateAnchors();

// ── PNG pixel extraction ──────────────────────────────────────────────────────
export function readUint32BE(bytes: Uint8Array, offset: number): number {
  return (
    ((bytes[offset] << 24) |
      (bytes[offset + 1] << 16) |
      (bytes[offset + 2] << 8) |
      bytes[offset + 3]) >>>
    0
  );
}

/** Apply one PNG filter row, writing result into `out`. */
export function applyPNGFilter(
  filterType: number,
  raw: Uint8Array,
  prev: Uint8Array,
  out: Uint8Array,
  bpp: number
): void {
  const n = raw.length;
  switch (filterType) {
    case 0: // None
      out.set(raw);
      break;
    case 1: // Sub
      for (let i = 0; i < n; i++) {
        out[i] = (raw[i] + (i >= bpp ? out[i - bpp] : 0)) & 0xff;
      }
      break;
    case 2: // Up
      for (let i = 0; i < n; i++) {
        out[i] = (raw[i] + prev[i]) & 0xff;
      }
      break;
    case 3: // Average
      for (let i = 0; i < n; i++) {
        const a = i >= bpp ? out[i - bpp] : 0;
        out[i] = (raw[i] + Math.floor((a + prev[i]) / 2)) & 0xff;
      }
      break;
    case 4: { // Paeth
      for (let i = 0; i < n; i++) {
        const a = i >= bpp ? out[i - bpp] : 0;
        const b = prev[i];
        const c = i >= bpp ? prev[i - bpp] : 0;
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        const paeth = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
        out[i] = (raw[i] + paeth) & 0xff;
      }
      break;
    }
  }
}

/**
 * Decode a PNG Uint8Array (from base64) into a flat Uint8Array of pixel bytes.
 * Returns interleaved [R, G, B, R, G, B, ...] (RGB) or [R, G, B, A, ...] (RGBA).
 * Requires `pako` for DEFLATE decompression.
 */
export function decodePNG(
  pngBytes: Uint8Array
): { pixels: Uint8Array; width: number; height: number; channels: number } {
  let offset = 8; // skip PNG signature
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
      pngBytes[offset + 3]
    );
    offset += 4;
    const data = pngBytes.slice(offset, offset + length);
    offset += length + 4; // +4 for CRC

    if (type === "IHDR") {
      width = readUint32BE(data, 0);
      height = readUint32BE(data, 4);
      // bitDepth = data[8] (assumed 8)
      const colorType = data[9]; // 2=RGB, 6=RGBA
      channels = colorType === 6 ? 4 : 3;
    } else if (type === "IDAT") {
      idatChunks.push(data);
    } else if (type === "IEND") {
      break;
    }
  }

  // Concatenate IDAT chunks and INFLATE
  const totalLength = idatChunks.reduce((s, c) => s + c.length, 0);
  const idat = new Uint8Array(totalLength);
  let pos = 0;
  for (const chunk of idatChunks) {
    idat.set(chunk, pos);
    pos += chunk.length;
  }
  const raw = pako.inflate(idat) as Uint8Array;

  // Reconstruct pixels applying PNG row filters
  const bpp = channels;
  const rowBytes = width * channels;
  const pixels = new Uint8Array(width * height * channels);
  const prevRow = new Uint8Array(rowBytes); // zero for first row

  for (let y = 0; y < height; y++) {
    const rowStart = y * (rowBytes + 1);
    const filterType = raw[rowStart];
    const rowData = raw.slice(rowStart + 1, rowStart + 1 + rowBytes);
    const outRow = pixels.subarray(y * rowBytes, (y + 1) * rowBytes);
    const prevRowSlice =
      y > 0 ? pixels.subarray((y - 1) * rowBytes, y * rowBytes) : prevRow;
    applyPNGFilter(filterType, rowData, prevRowSlice, outRow, bpp);
  }

  return { pixels, width, height, channels };
}

/**
 * Resize an image to `size × size` and return a Float32Array in CHW format,
 * normalized to [-1, 1]. Used for BlazeFace preprocessing.
 *
 * Requires `pako` (npm install pako).
 */
export async function imageUriToTensor(
  uri: string,
  size: number
): Promise<Float32Array> {
  // 1. Resize to target dimensions, export as base64 PNG
  const resized = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width: size, height: size } }],
    { format: ImageManipulator.SaveFormat.PNG, base64: true }
  );

  if (!resized.base64) throw new Error("imageUriToTensor: no base64 returned");

  // 2. Decode base64 → Uint8Array
  const binaryStr = atob(resized.base64);
  const pngBytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) {
    pngBytes[i] = binaryStr.charCodeAt(i);
  }

  // 3. Decode PNG → raw pixels [R,G,B,...] or [R,G,B,A,...]
  const { pixels, channels } = decodePNG(pngBytes);

  // 4. Build Float32Array in CHW layout, normalized to [-1, 1]
  const numPixels = size * size;
  const tensor = new Float32Array(3 * numPixels);

  for (let i = 0; i < numPixels; i++) {
    tensor[i] = pixels[i * channels] / 127.5 - 1.0;                     // R
    tensor[numPixels + i] = pixels[i * channels + 1] / 127.5 - 1.0;    // G
    tensor[2 * numPixels + i] = pixels[i * channels + 2] / 127.5 - 1.0; // B
  }

  return tensor;
}

/**
 * Resize an image to 224×224 and return a Float32Array in CHW format,
 * normalized with ImageNet mean/std. Used for ViT age/gender inference.
 */
export async function imageUriToViTTensor(uri: string): Promise<Float32Array> {
  const resized = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width: VIT_INPUT_SIZE, height: VIT_INPUT_SIZE } }],
    { format: ImageManipulator.SaveFormat.PNG, base64: true }
  );
  if (!resized.base64) throw new Error("imageUriToViTTensor: no base64 returned");
  console.log("Resized image base64:", resized.uri);

  const binaryStr = atob(resized.base64);
  const pngBytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) pngBytes[i] = binaryStr.charCodeAt(i);

  const { pixels, channels } = decodePNG(pngBytes);

  const numPixels = VIT_INPUT_SIZE * VIT_INPUT_SIZE;
  const tensor = new Float32Array(3 * numPixels);

  for (let i = 0; i < numPixels; i++) {
    tensor[i]                 = (pixels[i * channels]     / 255 - VIT_MEAN[0]) / VIT_STD[0]; // R
    tensor[numPixels + i]     = (pixels[i * channels + 1] / 255 - VIT_MEAN[1]) / VIT_STD[1]; // G
    tensor[2 * numPixels + i] = (pixels[i * channels + 2] / 255 - VIT_MEAN[2]) / VIT_STD[2]; // B
  }
  return tensor;
}

/**
 * Run softmax on raw logits and return top-1 label + score.
 */
export function topFromLogits(logits: Float32Array, labels: string[]): { label: string; score: number } {
  // Softmax
  const max = Math.max(...Array.from(logits));
  const exps = Array.from(logits).map((v) => Math.exp(v - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sum);

  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIdx]) bestIdx = i;
  }
  return { label: labels[bestIdx] ?? `class_${bestIdx}`, score: probs[bestIdx] };
}

/**
 * Returns all labels + scores sorted descending.
 */
export function allFromLogits(logits: Float32Array, labels: string[]): { label: string; score: number }[] {
  // Softmax
  const max = Math.max(...Array.from(logits));
  const exps = Array.from(logits).map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sum);

  // Combine labels and probabilities
  const results = probs.map((score, idx) => ({
    label: labels[idx] ?? `class_${idx}`,
    score,
  }));

  // Sort descending by score
  results.sort((a, b) => b.score - a.score);

  return results;
}

// ── BlazeFace postprocessing ──────────────────────────────────────────────────

/**
 * Decode raw network output into corner-format bounding boxes.
 * Matches BlazeFace._decode_boxes() from blazeface.py exactly:
 *   x_center = raw[...,0] / x_scale * anchor_w + anchor_cx
 *   y_center = raw[...,1] / y_scale * anchor_h + anchor_cy
 *   w        = raw[...,2] / w_scale * anchor_w
 *   h        = raw[...,3] / h_scale * anchor_h
 * x_scale = y_scale = w_scale = h_scale = 128.0 (front model)
 */
export function decodeBoxes(
  rawBoxes: Float32Array, // [896 * 16] flattened
  anchors: number[][]    // [896][cx, cy, aw, ah]  (aw=ah=1.0)
): number[][] {
  const boxes: number[][] = [];
  for (let i = 0; i < BLAZEFACE_NUM_ANCHORS; i++) {
    const [anchorCx, anchorCy, anchorW, anchorH] = anchors[i];
    const xCenter = rawBoxes[i * 16 + 0] / BLAZEFACE_INPUT_SIZE * anchorW + anchorCx;
    const yCenter = rawBoxes[i * 16 + 1] / BLAZEFACE_INPUT_SIZE * anchorH + anchorCy;
    const w       = rawBoxes[i * 16 + 2] / BLAZEFACE_INPUT_SIZE * anchorW;
    const h       = rawBoxes[i * 16 + 3] / BLAZEFACE_INPUT_SIZE * anchorH;
    boxes.push([yCenter - h / 2, xCenter - w / 2, yCenter + h / 2, xCenter + w / 2]);
  }
  return boxes;
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function iou(a: number[], b: number[]): number {
  const interY1 = Math.max(a[0], b[0]);
  const interX1 = Math.max(a[1], b[1]);
  const interY2 = Math.min(a[2], b[2]);
  const interX2 = Math.min(a[3], b[3]);
  const interArea =
    Math.max(0, interY2 - interY1) * Math.max(0, interX2 - interX1);
  const aArea = (a[2] - a[0]) * (a[3] - a[1]);
  const bArea = (b[2] - b[0]) * (b[3] - b[1]);
  return interArea / (aArea + bArea - interArea + 1e-6);
}

/** Non-maximum suppression — returns indices of kept boxes. */
export function nms(boxes: number[][], scores: number[], iouThreshold: number): number[] {
  const order = scores
    .map((s, i) => ({ s, i }))
    .sort((a, b) => b.s - a.s)
    .map((x) => x.i);

  const suppressed = new Set<number>();
  const keep: number[] = [];

  for (const i of order) {
    if (suppressed.has(i)) continue;
    keep.push(i);
    for (const j of order) {
      if (j === i || suppressed.has(j)) continue;
      if (iou(boxes[i], boxes[j]) > iouThreshold) suppressed.add(j);
    }
  }
  return keep;
}

/**
 * Run full BlazeFace postprocessing: score filter → anchor decode → NMS.
 * Returns bounding boxes in normalized [0,1] coords as [ymin, xmin, ymax, xmax].
 */
export function postprocessBlazeFace(outputs: TensorPtr[]): BBox[] {
  // forward() returns (boxes [1,896,16], scores [1,896,1]) — matches blazeface.py return [r, c]
  if (outputs.length < 2) {
    console.warn("BlazeFace: expected 2 output tensors, got", outputs.length);
    return [];
  }

  const rawBoxes  = new Float32Array(outputs[0].dataPtr as ArrayBuffer); // [896 * 16]
  const rawScores = new Float32Array(outputs[1].dataPtr as ArrayBuffer); // [896]

  // Filter by score threshold
  const scores: number[] = [];
  const candidates: number[] = [];
  for (let i = 0; i < BLAZEFACE_NUM_ANCHORS; i++) {
    const score = sigmoid(rawScores[i]);
    if (score >= SCORE_THRESHOLD) {
      scores.push(score);
      candidates.push(i);
    }
  }

  if (candidates.length === 0) return [];

  // Decode all candidate boxes
  const allBoxes = decodeBoxes(rawBoxes, ANCHORS);
  const candidateBoxes = candidates.map((i) => allBoxes[i]);
  const candidateScores = candidates.map((i) => sigmoid(rawScores[i]));

  // NMS
  const kept = nms(candidateBoxes, candidateScores, NMS_IOU_THRESHOLD);

  return kept.map((k) => {
    const [ymin, xmin, ymax, xmax] = candidateBoxes[k];
    return {
      ymin: Math.max(0, ymin),
      xmin: Math.max(0, xmin),
      ymax: Math.min(1, ymax),
      xmax: Math.min(1, xmax),
    };
  });
}

/** Crop a face from a photo URI, adding 20% padding around the bbox. */
export async function cropFace(photoUri: string, bbox: BBox, photoW: number, photoH: number): Promise<string> {
  const pad_x = (bbox.xmax - bbox.xmin) * 0.2;
  const pad_y = (bbox.ymax - bbox.ymin) * 0.2;
  const x1 = Math.max(0, (bbox.xmin - pad_x) * photoW);
  const y1 = Math.max(0, (bbox.ymin - pad_y) * photoH);
  const x2 = Math.min(photoW, (bbox.xmax + pad_x) * photoW);
  const y2 = Math.min(photoH, (bbox.ymax + pad_y) * photoH);

  const result = await ImageManipulator.manipulateAsync(
    photoUri,
    [{ crop: { originX: x1, originY: y1, width: x2 - x1, height: y2 - y1 } }],
    { format: ImageManipulator.SaveFormat.JPEG, compress: 0.9 }
  );
  return result.uri;
}

// ── Anchor generation ─────────────────────────────────────────────────────────
// Deterministic BlazeFace anchors for 128×128 input (matches anchors.npy exactly).
// Layer 0: 16×16 feature map, stride 8, 2 anchors/cell  → 512 anchors
// Layer 1:  8×8  feature map, stride 16, 6 anchors/cell → 384 anchors
// Total: 896 anchors, each [cx, cy, 1.0, 1.0] in normalized [0,1] coords
export function generateAnchors(): number[][] {
  const anchors: number[][] = [];

  // Layer 0
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      for (let k = 0; k < 2; k++) {
        anchors.push([(x + 0.5) / 16, (y + 0.5) / 16, 1.0, 1.0]);
      }
    }
  }

  // Layer 1
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      for (let k = 0; k < 6; k++) {
        anchors.push([(x + 0.5) / 8, (y + 0.5) / 8, 1.0, 1.0]);
      }
    }
  }

  return anchors; // 896 rows
}
