import { createContext, useContext, useEffect, useState } from "react";
import { Linking } from "react-native";
import * as MediaLibrary from "expo-media-library";

import { useAnalysisJob } from "@/hooks/useAnalysisJob";
import { estimateAnalysisDuration } from "@/services/analysisPipeline";
import { loadLibraryAssets } from "@/services/mediaLibrary";
import {
  INITIAL_JOB_STATE,
  type AnalysisConfig,
  type AnalysisJobState,
  type AssetJobProgress,
  type ImageAnalysisResult,
  type ModelStatus,
  type PermissionState,
} from "@/types/workflow";

type WorkflowContextValue = {
  libraryAssets: MediaLibrary.Asset[];
  jobAssets: MediaLibrary.Asset[];
  selectedAssetIds: string[];
  selectedAssets: MediaLibrary.Asset[];
  permissionState: PermissionState;
  analysisConfig: AnalysisConfig;
  jobState: AnalysisJobState;
  jobProgressByAsset: Record<string, AssetJobProgress>;
  resultsByAsset: Record<string, ImageAnalysisResult>;
  modelStatuses: ModelStatus[];
  modelsReady: boolean;
  overallModelProgress: number;
  isLibraryLoading: boolean;
  libraryError: string | null;
  estimatedDurationSeconds: number;
  toggleAssetSelection: (assetId: string) => void;
  clearSelection: () => void;
  requestLibraryPermission: () => Promise<void>;
  presentPermissionsPicker: () => Promise<void>;
  openSystemSettings: () => Promise<void>;
  refreshLibrary: () => Promise<void>;
  setAnalysisMode: (mode: AnalysisConfig["mode"]) => void;
  startJob: () => Promise<void>;
  cancelJob: () => void;
  retryLastJob: () => Promise<void>;
};

const WorkflowContext = createContext<WorkflowContextValue | null>(null);

function toPermissionState(
  permissionResponse: MediaLibrary.PermissionResponse | null | undefined,
): PermissionState {
  return {
    accessPrivileges: permissionResponse?.accessPrivileges ?? null,
    canAskAgain: permissionResponse?.canAskAgain ?? true,
    isGranted: permissionResponse?.status === "granted",
    isLimited: permissionResponse?.accessPrivileges === "limited",
    status: permissionResponse?.status ?? null,
  };
}

function normalizeError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

export function WorkflowProvider({ children }: { children: React.ReactNode }) {
  const [permissionResponse, requestPermission] = MediaLibrary.usePermissions();
  const [libraryAssets, setLibraryAssets] = useState<MediaLibrary.Asset[]>([]);
  const [jobAssets, setJobAssets] = useState<MediaLibrary.Asset[]>([]);
  const [selectedAssetIds, setSelectedAssetIds] = useState<string[]>([]);
  const [analysisConfig, setAnalysisConfig] = useState<AnalysisConfig>({ mode: "full" });
  const [jobState, setJobState] = useState<AnalysisJobState>(INITIAL_JOB_STATE);
  const [jobProgressByAsset, setJobProgressByAsset] = useState<Record<string, AssetJobProgress>>({});
  const [resultsByAsset, setResultsByAsset] = useState<Record<string, ImageAnalysisResult>>({});
  const [isLibraryLoading, setIsLibraryLoading] = useState(false);
  const [libraryError, setLibraryError] = useState<string | null>(null);

  const permissionState = toPermissionState(permissionResponse);
  const selectedAssets = libraryAssets.filter((asset) => selectedAssetIds.includes(asset.id));

  const { cancelJob, modelStatuses, modelsReady, overallModelProgress, retryLastJob, startJob } =
    useAnalysisJob({
      analysisConfig,
      selectedAssets,
      setJobAssets,
      setJobProgressByAsset,
      setJobState,
      setResultsByAsset,
    });

  const refreshLibrary = async (): Promise<void> => {
    if (!permissionState.isGranted) {
      setLibraryAssets([]);
      setSelectedAssetIds([]);
      setLibraryError(null);
      return;
    }

    setIsLibraryLoading(true);
    setLibraryError(null);

    try {
      const assets = await loadLibraryAssets();

      setLibraryAssets(assets);
      setSelectedAssetIds((current) => current.filter((assetId) => assets.some((asset) => asset.id === assetId)));
    } catch (error) {
      setLibraryError(normalizeError(error));
    } finally {
      setIsLibraryLoading(false);
    }
  };

  useEffect(() => {
    if (permissionState.isGranted) {
      void refreshLibrary();
      return;
    }

    if (permissionState.status === "denied") {
      setLibraryAssets([]);
      setSelectedAssetIds([]);
    }
  }, [permissionState.accessPrivileges, permissionState.isGranted, permissionState.status]);

  const requestLibraryPermission = async (): Promise<void> => {
    await requestPermission();
  };

  const presentPermissionsPicker = async (): Promise<void> => {
    if (permissionState.isGranted) {
      await MediaLibrary.presentPermissionsPickerAsync();
      await refreshLibrary();
      return;
    }

    await requestPermission();
  };

  const openSystemSettings = async (): Promise<void> => {
    await Linking.openSettings();
  };

  const toggleAssetSelection = (assetId: string): void => {
    setSelectedAssetIds((current) =>
      current.includes(assetId)
        ? current.filter((currentId) => currentId !== assetId)
        : [...current, assetId],
    );
  };

  const clearSelection = (): void => {
    setSelectedAssetIds([]);
  };

  const setAnalysisMode = (mode: AnalysisConfig["mode"]): void => {
    setAnalysisConfig((current) => ({
      ...current,
      mode,
    }));
  };

  return (
    <WorkflowContext.Provider
      value={{
        analysisConfig,
        cancelJob,
        clearSelection,
        estimatedDurationSeconds: estimateAnalysisDuration(selectedAssetIds.length, analysisConfig.mode),
        isLibraryLoading,
        jobProgressByAsset,
        jobState,
        jobAssets,
        libraryAssets,
        libraryError,
        modelStatuses,
        modelsReady,
        openSystemSettings,
        overallModelProgress,
        permissionState,
        presentPermissionsPicker,
        refreshLibrary,
        requestLibraryPermission,
        resultsByAsset,
        retryLastJob,
        selectedAssetIds,
        selectedAssets,
        setAnalysisMode,
        startJob,
        toggleAssetSelection,
      }}
    >
      {children}
    </WorkflowContext.Provider>
  );
}

export function useWorkflow(): WorkflowContextValue {
  const context = useContext(WorkflowContext);

  if (!context) {
    throw new Error("useWorkflow must be used within WorkflowProvider");
  }

  return context;
}
