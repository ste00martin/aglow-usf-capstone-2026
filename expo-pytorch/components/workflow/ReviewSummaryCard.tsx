import { Image, StyleSheet, Text, View, useWindowDimensions } from "react-native";
import type * as MediaLibrary from "expo-media-library";

import { workflowTheme } from "@/components/workflow/theme";
import { ANALYSIS_MODE_COPY, type AnalysisMode } from "@/types/workflow";

type ReviewSummaryCardProps = {
  estimatedDurationSeconds: number;
  mode: AnalysisMode;
  selectedAssets: MediaLibrary.Asset[];
};

export function ReviewSummaryCard({
  estimatedDurationSeconds,
  mode,
  selectedAssets,
}: ReviewSummaryCardProps) {
  const { width } = useWindowDimensions();
  const isCompact = width < 420;

  return (
    <View style={styles.card}>
      <Text style={styles.title}>Review summary</Text>
      <View style={[styles.statRow, isCompact ? styles.statRowCompact : undefined]}>
        <View style={[styles.stat, isCompact ? styles.statCompact : undefined]}>
          <Text style={styles.statValue}>{selectedAssets.length}</Text>
          <Text style={styles.statLabel}>photos</Text>
        </View>
        <View style={[styles.stat, isCompact ? styles.statCompact : undefined]}>
          <Text style={styles.statValue}>{Math.max(1, Math.round(estimatedDurationSeconds / 60))}m</Text>
          <Text style={styles.statLabel}>estimate</Text>
        </View>
        <View style={[styles.stat, isCompact ? styles.statCompact : undefined]}>
          <Text style={styles.statValue}>{ANALYSIS_MODE_COPY[mode].title}</Text>
          <Text style={styles.statLabel}>mode</Text>
        </View>
      </View>
      <Text style={styles.body}>{ANALYSIS_MODE_COPY[mode].subtitle}</Text>
      <View style={styles.previewRow}>
        {selectedAssets.slice(0, 4).map((asset) => (
          <Image key={asset.id} source={{ uri: asset.uri }} style={styles.previewImage} />
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  body: {
    color: workflowTheme.inkMuted,
    fontSize: 14,
    lineHeight: 20,
  },
  card: {
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 24,
    borderWidth: 1,
    gap: 14,
    padding: 18,
  },
  previewImage: {
    aspectRatio: 1,
    borderRadius: 16,
    width: 64,
  },
  previewRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  stat: {
    flex: 1,
    gap: 3,
    minWidth: 84,
  },
  statCompact: {
    minWidth: 120,
  },
  statLabel: {
    color: workflowTheme.inkMuted,
    fontSize: 12,
    textTransform: "uppercase",
  },
  statRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  statRowCompact: {
    rowGap: 12,
  },
  statValue: {
    color: workflowTheme.ink,
    fontSize: 18,
    flexShrink: 1,
    fontWeight: "700",
  },
  title: {
    color: workflowTheme.ink,
    fontSize: 19,
    fontWeight: "700",
  },
});
