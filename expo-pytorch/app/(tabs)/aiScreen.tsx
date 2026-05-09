/**
 * aiScreen.tsx — Face detection + demographics pipeline via ExecuTorch
 *
 * Pipeline per photo:
 *   1. BlazeFace (.pte) detects face bounding boxes  [useExecutorchModule]
 *   2. expo-image-manipulator crops each detected face
 *   3. Age classifier (.pte) runs on each crop       [useExecutorchModule + ViT preprocessing]
 *   4. Gender classifier (.pte) runs on each crop    [useExecutorchModule + ViT preprocessing]
 *
 * SETUP:
 *   1. For iOS/CoreML, run scripts/export_models_ios_coreml.sh
 *   2. For XNNPACK/CPU, run scripts/export_models_xnnpack.sh
 *   3. The scripts write the expected .pte files into assets/models/
 */

import {
  Text,
  View,
  StyleSheet,
  ScrollView,
  Pressable,
  Image as RNImage,
  ActivityIndicator,
} from "react-native";
import { Image } from 'expo-image';
import { useState, useContext, useEffect, useRef, useCallback, useLayoutEffect } from "react";
import { useNavigation } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import { useExecutorchModule, ScalarType } from "react-native-executorch";
import type { TensorPtr } from "react-native-executorch";
import { AlbumContext, type AlbumAsset } from "../../AlbumContext";
import { imageUriToTensor, postprocessBlazeFace, cropFace, imageUriToViTTensor, topFromLogits, allFromLogits } from "../../aipreprocessing" // for type-only imports of BBox, etc.
import {
  BLAZEFACE_MODEL,
  AGE_MODEL,
  GENDER_MODEL,
  NSFW_MODEL,
} from "../../assets/models/executorchModels";

// ── Model sources ─────────────────────────────────────────────────────────────
// Platform-specific requires live in executorchModels.ios.ts vs executorchModels.ts so Metro
// does not resolve missing sibling assets (e.g. XNNPACK .pte on iOS-only workflows).

// ── BlazeFace constants ───────────────────────────────────────────────────────
const BLAZEFACE_INPUT_SIZE = 128;

// ── ViT constants (age + gender models) ──────────────────────────────────────
const VIT_INPUT_SIZE = 224;
// ImageNet normalization: mean and std per channel
// Label sets from HuggingFace model configs
const AGE_LABELS = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', 'more than 70'];
const GENDER_LABELS = ['female', 'male'];
const NSFW_LABELS = ['gore_bloodshed_violent', 'nudity_pornography', 'safe_normal'];

// ── Types ─────────────────────────────────────────────────────────────────────
type BBox = { ymin: number; xmin: number; ymax: number; xmax: number };

type FaceResult = {
  bbox: BBox;
  faceURI: string;
  age: { label: string; score: number };
  gender: { label: string; score: number };
  nsfw: { label: string; score: number }[];
};

type ImageResult = {
  uri: string;
  faces: FaceResult[];
  error?: string;
};

function photoAssets(assets: AlbumAsset[]): AlbumAsset[] {
  return assets.filter((a) => a.mediaType === "photo");
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function AiScreen() {
  const navigation = useNavigation();
  const { assets } = useContext(AlbumContext);
  const assetsRef = useRef(assets);
  assetsRef.current = assets;

  const [imageResults, setImageResults] = useState<ImageResult[]>([]);
  /** True only while the async pipeline is executing (not used to choose which screen to show). */
  const [isProcessing, setIsProcessing] = useState(false);
  const isProcessingRef = useRef(false);
  isProcessingRef.current = isProcessing;
  const forceRerunAfterBatchRef = useRef(false);

  const faceDetector = useExecutorchModule({ modelSource: BLAZEFACE_MODEL });
  const ageModel = useExecutorchModule({ modelSource: AGE_MODEL });
  const genderModel = useExecutorchModule({ modelSource: GENDER_MODEL });
  const nsfwModel = useExecutorchModule({ modelSource: NSFW_MODEL });

  const isReady =
    faceDetector.isReady &&
    ageModel.isReady &&
    genderModel.isReady &&
    nsfwModel.isReady;

  const requestReplay = useCallback(() => {
    if (isProcessingRef.current) {
      forceRerunAfterBatchRef.current = true;
    } else {
      setIsProcessing(true);
    }
  }, []);

  const photoCount = photoAssets(assets).length;

  useLayoutEffect(() => {
    const hasCompletedRun = imageResults.length > 0;
    const canReplay =
      isReady && photoCount > 0 && !isProcessing && hasCompletedRun;
    const showReplayInHeader =
      isReady && !isProcessing && hasCompletedRun && photoCount > 0;

    navigation.setOptions({
      headerRight: showReplayInHeader
        ? () => (
            <Pressable
              onPress={requestReplay}
              hitSlop={12}
              style={styles.headerReplayBtn}
              disabled={!canReplay}
              accessibilityRole="button"
              accessibilityLabel="Replay pipeline on selected images"
            >
              <Ionicons name="reload" size={22} color="#fff" />
            </Pressable>
          )
        : undefined,
    });
  }, [
    navigation,
    requestReplay,
    isReady,
    photoCount,
    isProcessing,
    imageResults.length,
  ]);

  // faceDetector, ageModel, genderModel, nsfwModel omitted from deps — listing them restarts the effect every render.
  useEffect(() => {
    if (!isProcessing) return;
    if (!isReady) {
      setIsProcessing(false);
      return;
    }

    // Freeze selection for this run so picks added on the Images tab mid-pipeline are ignored until replay.
    const snapshotAtRunStart = photoAssets(assetsRef.current);

    let cancelled = false;

    const processOnePhoto = async (asset: AlbumAsset, results: ImageResult[]) => {
      const photoUri = asset.uri;

      try {
        const inputTensor = await imageUriToTensor(photoUri, BLAZEFACE_INPUT_SIZE);

        const inputTensorPtr: TensorPtr = {
          dataPtr: inputTensor,
          sizes: [1, 3, BLAZEFACE_INPUT_SIZE, BLAZEFACE_INPUT_SIZE],
          scalarType: ScalarType.FLOAT,
        };

        const detectorOutputs = await faceDetector.forward([inputTensorPtr]);
        const bboxes = postprocessBlazeFace(detectorOutputs);

        if (bboxes.length === 0) {
          results.push({ uri: photoUri, faces: [] });
          setImageResults([...results]);
          return;
        }

        const faces: FaceResult[] = [];
        const nonCroppedVitTensor = await imageUriToViTTensor(photoUri);
        const nonCroppedVitTensorPtr: TensorPtr = {
          dataPtr: nonCroppedVitTensor,
          sizes: [1, 3, VIT_INPUT_SIZE, VIT_INPUT_SIZE],
          scalarType: ScalarType.FLOAT,
        };
        const nsfwOutputs = await nsfwModel.forward([nonCroppedVitTensorPtr]);
        const nsfwLogits = new Float32Array(nsfwOutputs[0].dataPtr as ArrayBuffer);
        const nsfwScores = allFromLogits(nsfwLogits, NSFW_LABELS);

        let photoW = asset.width ?? 0;
        let photoH = asset.height ?? 0;
        if (photoW <= 0 || photoH <= 0) {
          try {
            const sz = await new Promise<{ w: number; h: number }>((resolve, reject) => {
              RNImage.getSize(photoUri, (w, h) => resolve({ w, h }), reject);
            });
            photoW = sz.w;
            photoH = sz.h;
          } catch {
            results.push({
              uri: photoUri,
              faces: [],
              error: "Could not read image dimensions for cropping.",
            });
            setImageResults([...results]);
            return;
          }
        }

        for (const bbox of bboxes) {
          const croppedUri = await cropFace(photoUri, bbox, photoW, photoH);

          console.log("Cropped Face Result: ", croppedUri);

          const vitTensor = await imageUriToViTTensor(croppedUri);
          const vitTensorPtr: TensorPtr = {
            dataPtr: vitTensor,
            sizes: [1, 3, VIT_INPUT_SIZE, VIT_INPUT_SIZE],
            scalarType: ScalarType.FLOAT,
          };

          const [ageOutputs, genderOutputs] = await Promise.all([
            ageModel.forward([vitTensorPtr]),
            genderModel.forward([vitTensorPtr]),
          ]);

          const ageLogits = new Float32Array(ageOutputs[0].dataPtr as ArrayBuffer);
          const genderLogits = new Float32Array(genderOutputs[0].dataPtr as ArrayBuffer);

          faces.push({
            bbox,
            faceURI: croppedUri,
            age: topFromLogits(ageLogits, AGE_LABELS),
            gender: topFromLogits(genderLogits, GENDER_LABELS),
            nsfw: nsfwScores,
          });
        }

        results.push({ uri: photoUri, faces });
      } catch (e) {
        console.error(`Error processing ${photoUri}:`, e);
        results.push({ uri: photoUri, faces: [], error: String(e) });
      }

      setImageResults([...results]);
    };

    (async () => {
      let useFrozenSnapshot = true;
      try {
        while (!cancelled) {
          const batch = useFrozenSnapshot
            ? snapshotAtRunStart
            : photoAssets(assetsRef.current);
          useFrozenSnapshot = false;

          if (batch.length === 0) {
            setImageResults([]);
            break;
          }

          const results: ImageResult[] = [];

          for (const asset of batch) {
            if (cancelled) return;
            await processOnePhoto(asset, results);
          }

          if (cancelled) return;

          setImageResults(results);

          // Only replay (forced) runs another pass, using the current album for that pass.
          const forced = forceRerunAfterBatchRef.current;
          if (forced) {
            forceRerunAfterBatchRef.current = false;
            continue;
          }
          break;
        }
      } finally {
        setIsProcessing(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [isReady, isProcessing]);

  // ── Loading state ────────────────────────────────────────────────────────
  if (!isReady) {
    const downloading = [faceDetector, ageModel, genderModel, nsfwModel].find(
      (m) => m.downloadProgress < 1
    );
    return (
      <View style={styles.center}>
        <Text>
          {downloading
            ? `Downloading models… ${Math.round((downloading.downloadProgress ?? 0) * 100)}%`
            : "Loading models…"}
        </Text>
        {faceDetector.error && (
          <Text style={styles.errorText}>BlazeFace: {String(faceDetector.error)}</Text>
        )}
        {ageModel.error && (
          <Text style={styles.errorText}>Age model: {String(ageModel.error)}</Text>
        )}
        {genderModel.error && (
          <Text style={styles.errorText}>Gender model: {String(genderModel.error)}</Text>
        )}
        {nsfwModel.error && (
          <Text style={styles.errorText}>NSFW model: {String(nsfwModel.error)}</Text>
        )}
      </View>
    );
  }

  const showAnalyzePrompt = imageResults.length === 0 && !isProcessing;

  return (
    <ScrollView contentContainerStyle={styles.container}>
      {isProcessing ? (
        <View style={styles.runningBanner}>
          <ActivityIndicator color="#333" />
          <Text style={styles.runningText}>Running pipeline on selected images…</Text>
        </View>
      ) : null}

      {showAnalyzePrompt ? (
        <View style={styles.startSection}>
          <Pressable
            style={styles.buttonContainer}
            onPress={() => setIsProcessing(true)}
            disabled={photoCount === 0}
          >
            <Text style={styles.button}>Analyze Faces</Text>
          </Pressable>
          {photoCount === 0 ? (
            <Text style={styles.hintText}>Add photos on the Images tab first.</Text>
          ) : (
            <Text style={styles.hintText}>{photoCount} photo{photoCount !== 1 ? "s" : ""} selected.</Text>
          )}
        </View>
      ) : null}

      {imageResults.map(({ uri, faces, error }) => (
        <View key={uri} style={styles.imageBlock}>
          <Image source={{ uri }} style={styles.headImage}/>

          {error ? (
            <Text style={styles.errorText}>Error: {error}</Text>
          ) : faces.length === 0 ? (
            <Text style={styles.noFaceText}>No faces detected</Text>
          ) : (
            faces.map((face, idx) => (
              <View key={idx} style={styles.faceBlock}>
                <Image source={{ uri: face.faceURI }} style={styles.image} />
                <View style={styles.faceInfo}>
                  <Text style={styles.faceHeader}>Face {idx + 1}</Text>
                  <Text style={styles.noFaceText}>
                    <Text style={{ fontWeight: "500" }}>Age:</Text> Age: {face.age.label} ({(face.age.score * 100).toFixed(1)}%)
                  </Text>
                  <Text style={styles.noFaceText}>
                    <Text style={{ fontWeight: "500" }}>Gender:</Text> {face.gender.label} (
                    {(face.gender.score * 100).toFixed(1)}%)
                  </Text>
                  <Text style={styles.noFaceText}>
                    <Text style={{ fontWeight: "500" }}>NSFW:</Text> {face.nsfw.map(n => `${n.label} (${(n.score * 100).toFixed(1)}%)`).join(", ")}
                  </Text>
                </View>
              </View>
            ))
          )}
        </View>
      ))}
    </ScrollView>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  headerReplayBtn: {
    paddingLeft: 8,
    paddingVertical: 4,
    marginRight: 32,
  },
  container: {
    padding: 16,
  },
  imageBlock: {
    marginBottom: 24,
    borderBottomWidth: 1,
    borderColor: "#ccc",
    paddingBottom: 12,
  },
  headImage: {
    width: "100%",
    height: 150,
    borderRadius: 8,
    marginBottom: 8,
    resizeMode: "contain",
  },
  image: {
    width: 100,
    height: 100,
    borderRadius: 8,
    resizeMode: "contain",
  },
  faceInfo: {
    flex: 1, // takes remaining width, enabling text wrapping
  },
  faceBlock: {
    backgroundColor: "#ffffff",
    borderRadius: 6,
    padding: 10,
    marginTop: 6,
    alignItems: "flex-start",
    flexDirection: "row",
    gap: 8, // adds space between all direct children
  },

  faceHeader: {
    fontWeight: "bold",
    marginBottom: 4,
  },
  noFaceText: {
    color: "#000000",
    flexWrap: "wrap",
  },
  errorText: {
    color: "#c0392b",
    fontSize: 12,
    marginTop: 4,
  },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  button: {
    fontSize: 20,
    color: "#fff",
  },
  buttonContainer: {
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    backgroundColor: "#333",
    margin: 8,
    alignItems: "center",
  },
  startSection: {
    alignItems: "center",
    paddingVertical: 24,
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  hintText: {
    marginTop: 12,
    fontSize: 15,
    color: "#555",
    textAlign: "center",
    paddingHorizontal: 16,
  },
  runningBanner: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginBottom: 16,
    paddingVertical: 8,
  },
  runningText: {
    fontSize: 15,
    color: "#333",
    flex: 1,
  },
});
