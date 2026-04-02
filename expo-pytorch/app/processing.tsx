import { router } from "expo-router";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { useWorkflow } from "@/AlbumContext";
import { ProcessingQueueList } from "@/components/workflow/ProcessingQueueList";
import { ScreenShell } from "@/components/workflow/ScreenShell";
import { workflowTheme } from "@/components/workflow/theme";

export default function ProcessingScreen() {
  const { cancelJob, jobAssets, jobProgressByAsset, jobState, retryLastJob, resultsByAsset, selectedAssets } =
    useWorkflow();

  const hasResults = Object.keys(resultsByAsset).length > 0;
  const queueAssets = jobAssets.length ? jobAssets : selectedAssets;

  return (
    <ScreenShell
      backHref="/review"
      subtitle="The pipeline now reports progress per photo instead of collapsing execution and results into one screen."
      title="Processing"
    >
      <View style={styles.summaryCard}>
        <Text style={styles.summaryTitle}>
          {jobState.status === "idle"
            ? "No job is running."
            : `${jobState.completed} of ${jobState.total} photos processed`}
        </Text>
        <Text style={styles.summaryBody}>
          {jobState.error
            ? jobState.error
            : jobState.status === "completed"
              ? "Analysis finished. Open results to inspect each photo."
              : jobState.status === "cancelled"
                ? "The current run was cancelled."
                : "Safety scan, face detection, and demographic passes update below in real time."}
        </Text>
      </View>
      <ProcessingQueueList
        activeAssetId={jobState.activeAssetId}
        assets={queueAssets}
        progressByAsset={jobProgressByAsset}
      />
      <View style={styles.actions}>
        {(jobState.status === "queued" || jobState.status === "running") && (
          <Pressable style={styles.secondaryButton} onPress={cancelJob}>
            <Text style={styles.secondaryLabel}>Cancel run</Text>
          </Pressable>
        )}
        {(jobState.status === "failed" || jobState.status === "cancelled") && (
          <Pressable
            style={styles.secondaryButton}
            onPress={() => {
              void retryLastJob();
            }}
          >
            <Text style={styles.secondaryLabel}>Retry</Text>
          </Pressable>
        )}
        {hasResults && (
          <Pressable style={styles.primaryButton} onPress={() => router.push("/results")}>
            <Text style={styles.primaryLabel}>Open results</Text>
          </Pressable>
        )}
      </View>
    </ScreenShell>
  );
}

const styles = StyleSheet.create({
  actions: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 22,
  },
  primaryButton: {
    backgroundColor: workflowTheme.accent,
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  primaryLabel: {
    color: "#FFF",
    fontSize: 15,
    fontWeight: "700",
  },
  secondaryButton: {
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 16,
    borderWidth: 1,
    paddingHorizontal: 16,
    paddingVertical: 14,
  },
  secondaryLabel: {
    color: workflowTheme.ink,
    fontSize: 15,
    fontWeight: "700",
  },
  summaryBody: {
    color: workflowTheme.inkMuted,
    fontSize: 14,
    lineHeight: 20,
  },
  summaryCard: {
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 24,
    borderWidth: 1,
    gap: 6,
    marginBottom: 18,
    padding: 18,
  },
  summaryTitle: {
    color: workflowTheme.ink,
    fontSize: 20,
    fontWeight: "700",
  },
});
