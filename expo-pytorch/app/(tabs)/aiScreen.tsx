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
  Image,
  Pressable,
  Platform,
} from "react-native";
import { useState, useContext, useEffect } from "react";
import { useExecutorchModule, ScalarType } from "react-native-executorch";
import type { TensorPtr } from "react-native-executorch";
import { AlbumContext } from "../../AlbumContext";
import { imageUriToTensor, postprocessBlazeFace, cropFace, imageUriToViTTensor, topFromLogits, allFromLogits } from "../../aipreprocessing" // for type-only imports of BBox, etc.

// ── Model sources ─────────────────────────────────────────────────────────────
// Keep XNNPACK as the shared default asset set and load the CoreML variants on iOS.
const IS_IOS = Platform.OS === "ios";
const BLAZEFACE_MODEL = IS_IOS
  ? require("../../assets/models/blazeface_coreml.pte")
  : require("../../assets/models/blazeface.pte");
const AGE_MODEL = IS_IOS
  ? require("../../assets/models/age_model_coreml.pte")
  : require("../../assets/models/age_model.pte");
const GENDER_MODEL = IS_IOS
  ? require("../../assets/models/gender_model_coreml.pte")
  : require("../../assets/models/gender_model.pte");
const NSFW_MODEL = IS_IOS
  ? require("../../assets/models/nsfw_model_coreml.pte")
  : require("../../assets/models/nsfw_model.pte");

// ── BlazeFace constants ───────────────────────────────────────────────────────
const BLAZEFACE_INPUT_SIZE = 128;

// ── ViT constants (age + gender models) ──────────────────────────────────────
const VIT_INPUT_SIZE = 224;
// ImageNet normalization: mean and std per channel
// Label sets from HuggingFace model configs
const AGE_LABELS    = ['0-2','3-9','10-19','20-29','30-39','40-49','50-59','60-69','more than 70'];
const GENDER_LABELS = ['female','male'];
const NSFW_LABELS   = ['gore_bloodshed_violent', 'nudity_pornography', 'safe_normal'];

// ── Types ─────────────────────────────────────────────────────────────────────
type BBox = { ymin: number; xmin: number; ymax: number; xmax: number };

type FaceResult = {
  bbox: BBox;
  age: { label: string; score: number };
  gender: { label: string; score: number };
  nsfw: { label: string; score: number }[];
};

type ImageResult = {
  uri: string;
  faces: FaceResult[];
  error?: string;
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function AiScreen() {
  const { assets } = useContext(AlbumContext);
  const [imageResults, setImageResults] = useState<ImageResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  const faceDetector = useExecutorchModule({ modelSource: BLAZEFACE_MODEL });
  const ageModel     = useExecutorchModule({ modelSource: AGE_MODEL });
  const genderModel  = useExecutorchModule({ modelSource: GENDER_MODEL });
  const nsfwModel = useExecutorchModule({ modelSource: NSFW_MODEL });

  const isReady =
    faceDetector.isReady &&
    ageModel.isReady &&
    genderModel.isReady &&
    nsfwModel.isReady;

  useEffect(() => {
    if (!isReady || assets.length === 0 || !isRunning) return;

    const runPipeline = async () => {
      const results: ImageResult[] = [];

      for (const asset of assets) {
        if (asset.mediaType !== "photo"){
          continue;
        }
        const photoUri = asset.uri;

        try {
          // ── Step 1: BlazeFace detection ──────────────────────────────────
          const inputTensor = await imageUriToTensor(
            photoUri,
            BLAZEFACE_INPUT_SIZE
          );

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
            continue;
          }

          // ── Step 2–4: Crop → age + gender per face ───────────────────────
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

          for (const bbox of bboxes) {
            const croppedUri = await cropFace(
              photoUri,
              bbox,
              asset.width,
              asset.height
            );

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
              age:    topFromLogits(ageLogits,    AGE_LABELS),
              gender: topFromLogits(genderLogits, GENDER_LABELS),
              nsfw:   nsfwScores,
            });
          }

          results.push({ uri: photoUri, faces });
        } catch (e) {
          console.error(`Error processing ${photoUri}:`, e);
          results.push({ uri: photoUri, faces: [], error: String(e) });
        }

        setImageResults([...results]);
      }
    };

    runPipeline();
  }, [isReady, assets, isRunning]);

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

  // ── Start button ─────────────────────────────────────────────────────────
  if (!isRunning) {
    return (
      <Pressable
        style={styles.buttonContainer}
        onPress={() => setIsRunning(true)}
      >
        <Text style={styles.button}>Analyze Faces</Text>
      </Pressable>
    );
  }

  // ── Results ───────────────────────────────────────────────────────────────
  return (
    <ScrollView contentContainerStyle={styles.container}>
      {imageResults.map(({ uri, faces, error }) => (
        <View key={uri} style={styles.imageBlock}>
          <Image source={{ uri }} style={styles.image} resizeMode="contain" />

          {error ? (
            <Text style={styles.errorText}>Error: {error}</Text>
          ) : faces.length === 0 ? (
            <Text style={styles.noFaceText}>No faces detected</Text>
          ) : (
            faces.map((face, idx) => (
              <View key={idx} style={styles.faceBlock}>
                <Text style={styles.faceHeader}>Face {idx + 1}</Text>
                <Text>
                  Age: {face.age.label} ({(face.age.score * 100).toFixed(1)}%)
                </Text>
                <Text>
                  Gender: {face.gender.label} (
                  {(face.gender.score * 100).toFixed(1)}%)
                </Text>
                <Text>
                  NSFW: {face.nsfw.map(n => `${n.label} (${(n.score*100).toFixed(1)}%)`).join(", ")}
                </Text>
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
  container: {
    padding: 16,
  },
  imageBlock: {
    marginBottom: 24,
    borderBottomWidth: 1,
    borderColor: "#ccc",
    paddingBottom: 12,
  },
  image: {
    width: "100%",
    height: 200,
    borderRadius: 8,
    marginBottom: 8,
  },
  faceBlock: {
    backgroundColor: "#f5f5f5",
    borderRadius: 6,
    padding: 8,
    marginTop: 6,
  },
  faceHeader: {
    fontWeight: "bold",
    marginBottom: 2,
  },
  noFaceText: {
    color: "#888",
    fontStyle: "italic",
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
});
