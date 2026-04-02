import { useEffect, useRef } from "react";
import * as MediaLibrary from "expo-media-library";
import { useExecutorchModule } from "react-native-executorch";

import {
  ANALYSIS_STAGE_LABELS,
  type AnalysisConfig,
  type AnalysisJobState,
  type AssetJobProgress,
  type ImageAnalysisResult,
  type ModelStatus,
} from "@/types/workflow";
import {
  AnalysisCancelledError,
  analyzeAsset,
  type AnalysisProgressEvent,
} from "@/services/analysisPipeline";

const BLAZEFACE_MODEL = require("../assets/models/blazeface.pte");
const AGE_MODEL = require("../assets/models/age_model.pte");
const GENDER_MODEL = require("../assets/models/gender_model.pte");
const NSFW_MODEL = require("../assets/models/nsfw_model.pte");
const NSFW_LABELS = require("../assets/models/nsfw_labels.json") as string[];

type UseAnalysisJobOptions = {
  analysisConfig: AnalysisConfig;
  selectedAssets: MediaLibrary.Asset[];
  setJobAssets: React.Dispatch<React.SetStateAction<MediaLibrary.Asset[]>>;
  setJobState: React.Dispatch<React.SetStateAction<AnalysisJobState>>;
  setJobProgressByAsset: React.Dispatch<
    React.SetStateAction<Record<string, AssetJobProgress>>
  >;
  setResultsByAsset: React.Dispatch<
    React.SetStateAction<Record<string, ImageAnalysisResult>>
  >;
};

type UseAnalysisJobResult = {
  modelStatuses: ModelStatus[];
  modelsReady: boolean;
  overallModelProgress: number;
  startJob: () => Promise<void>;
  retryLastJob: () => Promise<void>;
  cancelJob: () => void;
};

function toErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

export function useAnalysisJob(options: UseAnalysisJobOptions): UseAnalysisJobResult {
  const faceDetector = useExecutorchModule({ modelSource: BLAZEFACE_MODEL });
  const ageModel = useExecutorchModule({ modelSource: AGE_MODEL });
  const genderModel = useExecutorchModule({ modelSource: GENDER_MODEL });
  const nsfwModel = useExecutorchModule({ modelSource: NSFW_MODEL });

  const cancelRef = useRef(false);
  const activeRunIdRef = useRef(0);
  const lastSelectedAssetsRef = useRef<MediaLibrary.Asset[]>([]);

  const modelStatuses: ModelStatus[] = [
    {
      downloadProgress: faceDetector.downloadProgress ?? 0,
      error: faceDetector.error ? toErrorMessage(faceDetector.error) : null,
      isReady: faceDetector.isReady,
      key: "blazeface",
      label: "BlazeFace",
    },
    {
      downloadProgress: ageModel.downloadProgress ?? 0,
      error: ageModel.error ? toErrorMessage(ageModel.error) : null,
      isReady: ageModel.isReady,
      key: "age",
      label: "Age",
    },
    {
      downloadProgress: genderModel.downloadProgress ?? 0,
      error: genderModel.error ? toErrorMessage(genderModel.error) : null,
      isReady: genderModel.isReady,
      key: "gender",
      label: "Gender",
    },
    {
      downloadProgress: nsfwModel.downloadProgress ?? 0,
      error: nsfwModel.error ? toErrorMessage(nsfwModel.error) : null,
      isReady: nsfwModel.isReady,
      key: "nsfw",
      label: "Safety",
    },
  ];

  const modelsReady = modelStatuses.every((status) => status.isReady);
  const overallModelProgress =
    modelStatuses.reduce(
      (sum, status) => sum + (status.isReady ? 1 : status.downloadProgress),
      0,
    ) / modelStatuses.length;

  useEffect(() => {
    return () => {
      cancelRef.current = true;
    };
  }, []);

  const cancelJob = (): void => {
    cancelRef.current = true;
  };

  const markRemainingAssetsCancelled = (assets: MediaLibrary.Asset[]): void => {
    const cancelledAt = Date.now();

    options.setJobProgressByAsset((current) => {
      const next = { ...current };

      for (const asset of assets) {
        const existing = next[asset.id];

        if (!existing || existing.status === "queued") {
          next[asset.id] = {
            assetId: asset.id,
            error: undefined,
            progress: existing?.progress ?? 0,
            stageLabel: ANALYSIS_STAGE_LABELS.cancelled,
            status: "cancelled",
            updatedAt: cancelledAt,
            uri: asset.uri,
          };
        }
      }

      return next;
    });
  };

  const runJob = async (assets: MediaLibrary.Asset[]): Promise<void> => {
    if (!assets.length || !modelsReady) {
      return;
    }

    cancelRef.current = false;
    const runId = Date.now();
    activeRunIdRef.current = runId;
    lastSelectedAssetsRef.current = assets;

    options.setJobAssets(assets);
    options.setResultsByAsset({});
    options.setJobProgressByAsset(
      Object.fromEntries(
        assets.map((asset) => [
          asset.id,
          {
            assetId: asset.id,
            progress: 0,
            stageLabel: ANALYSIS_STAGE_LABELS.queued,
            status: "queued",
            updatedAt: Date.now(),
            uri: asset.uri,
          },
        ]),
      ),
    );
    options.setJobState({
      activeAssetId: assets[0]?.id ?? null,
      completed: 0,
      completedAt: null,
      error: null,
      runId,
      startedAt: Date.now(),
      status: "queued",
      total: assets.length,
    });

    let completedCount = 0;
    let failedCount = 0;

    for (let index = 0; index < assets.length; index += 1) {
      const asset = assets[index];

      if (cancelRef.current || activeRunIdRef.current !== runId) {
        markRemainingAssetsCancelled(assets.slice(index));
        options.setJobState((current) => ({
          ...current,
          activeAssetId: null,
          completed: completedCount,
          completedAt: Date.now(),
          status: "cancelled",
        }));
        return;
      }

      options.setJobState((current) => ({
        ...current,
        activeAssetId: asset.id,
        error: null,
        status: "running",
      }));

      const handleProgress = (event: AnalysisProgressEvent): void => {
        if (activeRunIdRef.current !== runId) {
          return;
        }

        options.setJobProgressByAsset((current) => ({
          ...current,
          [event.assetId]: {
            assetId: event.assetId,
            error: event.error,
            faceCount: event.faceCount,
            progress: event.progress,
            stageLabel: ANALYSIS_STAGE_LABELS[event.stage],
            status: event.stage,
            updatedAt: Date.now(),
            uri: event.uri,
          },
        }));
      };

      try {
        const result = await analyzeAsset({
          asset,
          mode: options.analysisConfig.mode,
          modules: {
            ageModelForward: ageModel.forward,
            faceDetectorForward: faceDetector.forward,
            genderModelForward: genderModel.forward,
            nsfwLabels: NSFW_LABELS,
            nsfwModelForward: nsfwModel.forward,
          },
          onProgress: handleProgress,
          shouldCancel: () => cancelRef.current || activeRunIdRef.current !== runId,
        });

        if (activeRunIdRef.current !== runId) {
          return;
        }

        options.setResultsByAsset((current) => ({
          ...current,
          [asset.id]: result,
        }));
      } catch (error) {
        if (error instanceof AnalysisCancelledError) {
          markRemainingAssetsCancelled(assets.slice(index));
          options.setJobState((current) => ({
            ...current,
            activeAssetId: null,
            completed: completedCount,
            completedAt: Date.now(),
            status: "cancelled",
          }));
          return;
        }

        failedCount += 1;
        const errorMessage = toErrorMessage(error);

        handleProgress({
          assetId: asset.id,
          error: errorMessage,
          progress: 1,
          stage: "failed",
          uri: asset.uri,
        });
        options.setResultsByAsset((current) => ({
          ...current,
          [asset.id]: {
            analyzedAt: Date.now(),
            assetId: asset.id,
            error: errorMessage,
            faces: [],
            nsfw: [],
            uri: asset.uri,
          },
        }));
      }

      completedCount += 1;
      options.setJobState((current) => ({
        ...current,
        completed: completedCount,
      }));
    }

    options.setJobState((current) => ({
      ...current,
      activeAssetId: null,
      completed: completedCount,
      completedAt: Date.now(),
      error: failedCount > 0 ? `${failedCount} photo(s) failed during analysis.` : null,
      status: failedCount > 0 ? "failed" : "completed",
    }));
  };

  const startJob = async (): Promise<void> => {
    await runJob(options.selectedAssets);
  };

  const retryLastJob = async (): Promise<void> => {
    await runJob(lastSelectedAssetsRef.current.length ? lastSelectedAssetsRef.current : options.selectedAssets);
  };

  return {
    cancelJob,
    modelStatuses,
    modelsReady,
    overallModelProgress,
    retryLastJob,
    startJob,
  };
}
