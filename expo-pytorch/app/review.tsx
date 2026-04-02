import { router } from "expo-router";
import { Image, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";

import { useWorkflow } from "@/AlbumContext";
import { AnalysisModeToggle } from "@/components/workflow/AnalysisModeToggle";
import { ModelStatusCard } from "@/components/workflow/ModelStatusCard";
import { ReviewSummaryCard } from "@/components/workflow/ReviewSummaryCard";
import { ScreenShell } from "@/components/workflow/ScreenShell";
import { workflowTheme } from "@/components/workflow/theme";

export default function ReviewScreen() {
  const {
    analysisConfig,
    estimatedDurationSeconds,
    modelStatuses,
    modelsReady,
    overallModelProgress,
    selectedAssets,
    setAnalysisMode,
    startJob,
  } = useWorkflow();

  if (!selectedAssets.length) {
    return (
      <ScreenShell
        backHref="/"
        subtitle="Select one or more photos in the workspace before starting a review run."
        title="Review"
      >
        <View style={styles.emptyCard}>
          <Text style={styles.emptyTitle}>No photos selected.</Text>
          <Pressable style={styles.primaryButton} onPress={() => router.replace("/")}>
            <Text style={styles.primaryLabel}>Back to workspace</Text>
          </Pressable>
        </View>
      </ScreenShell>
    );
  }

  return (
    <ScreenShell
      backHref="/"
      subtitle="Confirm the queue, check readiness, and start the current on-device pipeline."
      title="Review"
    >
      <ReviewSummaryCard
        estimatedDurationSeconds={estimatedDurationSeconds}
        mode={analysisConfig.mode}
        selectedAssets={selectedAssets}
      />
      <View style={styles.section}>
        <AnalysisModeToggle mode={analysisConfig.mode} onChange={setAnalysisMode} />
      </View>
      <View style={styles.section}>
        <ModelStatusCard
          modelStatuses={modelStatuses}
          modelsReady={modelsReady}
          overallModelProgress={overallModelProgress}
        />
      </View>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Queued photos</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.previewStrip}>
          <View style={styles.previewRow}>
            {selectedAssets.map((asset) => (
              <View key={asset.id} style={styles.previewCard}>
                <View style={styles.previewImageWrap}>
                  <Image source={{ uri: asset.uri }} style={styles.previewImage} />
                </View>
              </View>
            ))}
          </View>
        </ScrollView>
      </View>
      <Pressable
        disabled={!modelsReady}
        onPress={() => {
          void startJob();
          router.push("/processing");
        }}
        style={[styles.primaryButton, !modelsReady ? styles.buttonDisabled : undefined]}
      >
        <Text style={styles.primaryLabel}>{modelsReady ? "Start analysis" : "Waiting for models"}</Text>
      </Pressable>
    </ScreenShell>
  );
}

const styles = StyleSheet.create({
  buttonDisabled: {
    opacity: 0.5,
  },
  emptyCard: {
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 24,
    borderWidth: 1,
    gap: 14,
    padding: 18,
  },
  emptyTitle: {
    color: workflowTheme.ink,
    fontSize: 18,
    fontWeight: "700",
  },
  previewCard: {
    borderRadius: 18,
    overflow: "hidden",
  },
  previewImageWrap: {
    borderRadius: 18,
    height: 96,
    overflow: "hidden",
    width: 96,
  },
  previewImage: {
    height: "100%",
    width: "100%",
  },
  previewRow: {
    flexDirection: "row",
    gap: 10,
  },
  previewStrip: {
    marginTop: 12,
  },
  primaryButton: {
    backgroundColor: workflowTheme.accent,
    borderRadius: 18,
    marginTop: 24,
    paddingHorizontal: 18,
    paddingVertical: 16,
  },
  primaryLabel: {
    color: "#FFF",
    fontSize: 16,
    fontWeight: "700",
    textAlign: "center",
  },
  section: {
    marginTop: 18,
  },
  sectionTitle: {
    color: workflowTheme.ink,
    fontSize: 19,
    fontWeight: "700",
  },
});
