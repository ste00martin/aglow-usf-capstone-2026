import { Image, StyleSheet, Text, View } from "react-native";
import type * as MediaLibrary from "expo-media-library";

import { ProcessingStageBadge } from "@/components/workflow/ProcessingStageBadge";
import { workflowTheme } from "@/components/workflow/theme";
import type { AssetJobProgress } from "@/types/workflow";

type ProcessingQueueListProps = {
  activeAssetId: string | null;
  assets: MediaLibrary.Asset[];
  progressByAsset: Record<string, AssetJobProgress>;
};

export function ProcessingQueueList({
  activeAssetId,
  assets,
  progressByAsset,
}: ProcessingQueueListProps) {
  return (
    <View style={styles.list}>
      {assets.map((asset, index) => {
        const progress = progressByAsset[asset.id];

        return (
          <View key={asset.id} style={[styles.row, asset.id === activeAssetId ? styles.rowActive : undefined]}>
            <Image source={{ uri: asset.uri }} style={styles.image} />
            <View style={styles.copyWrap}>
              <Text style={styles.position}>Photo {index + 1}</Text>
              <Text style={styles.statusTitle}>{progress?.stageLabel ?? "Queued"}</Text>
              <View style={styles.badgeRow}>
                <ProcessingStageBadge
                  label={progress?.stageLabel ?? "Queued"}
                  stage={progress?.status ?? "queued"}
                />
                {typeof progress?.faceCount === "number" ? (
                  <Text style={styles.faceCount}>{progress.faceCount} faces</Text>
                ) : null}
              </View>
              <View style={styles.progressTrack}>
                <View
                  style={[
                    styles.progressFill,
                    { width: `${Math.max(6, Math.round((progress?.progress ?? 0) * 100))}%` },
                  ]}
                />
              </View>
              {progress?.error ? <Text style={styles.error}>{progress.error}</Text> : null}
            </View>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  badgeRow: {
    alignItems: "center",
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  copyWrap: {
    flex: 1,
    gap: 8,
  },
  error: {
    color: workflowTheme.danger,
    fontSize: 12,
    lineHeight: 18,
  },
  faceCount: {
    color: workflowTheme.inkMuted,
    fontSize: 12,
    fontWeight: "600",
  },
  image: {
    borderRadius: 16,
    height: 86,
    width: 86,
  },
  list: {
    gap: 12,
  },
  position: {
    color: workflowTheme.inkMuted,
    fontSize: 12,
    textTransform: "uppercase",
  },
  progressFill: {
    backgroundColor: workflowTheme.accent,
    borderRadius: 999,
    height: "100%",
  },
  progressTrack: {
    backgroundColor: workflowTheme.panelAlt,
    borderRadius: 999,
    height: 10,
    overflow: "hidden",
  },
  row: {
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 22,
    borderWidth: 1,
    flexDirection: "row",
    gap: 14,
    padding: 14,
  },
  rowActive: {
    borderColor: workflowTheme.accent,
  },
  statusTitle: {
    color: workflowTheme.ink,
    fontSize: 16,
    fontWeight: "700",
  },
});
