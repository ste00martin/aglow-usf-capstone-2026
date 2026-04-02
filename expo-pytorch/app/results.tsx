import { router } from "expo-router";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { useWorkflow } from "@/AlbumContext";
import { ResultImageCard } from "@/components/workflow/ResultImageCard";
import { ScreenShell } from "@/components/workflow/ScreenShell";
import { workflowTheme } from "@/components/workflow/theme";

export default function ResultsScreen() {
  const { jobAssets, resultsByAsset, selectedAssets } = useWorkflow();

  const sourceAssets = jobAssets.length ? jobAssets : selectedAssets;
  const orderedResults = sourceAssets
    .map((asset) => resultsByAsset[asset.id])
    .filter((result): result is NonNullable<typeof result> => Boolean(result));
  const faceCount = orderedResults.reduce((sum, result) => sum + result.faces.length, 0);
  const issueCount = orderedResults.filter((result) => result.error).length;

  if (!orderedResults.length) {
    return (
      <ScreenShell
        backHref="/processing"
        subtitle="Run the processing flow first to populate image cards and detail views."
        title="Results"
      >
        <View style={styles.emptyCard}>
          <Text style={styles.emptyTitle}>No results yet.</Text>
          <Pressable style={styles.primaryButton} onPress={() => router.replace("/")}>
            <Text style={styles.primaryLabel}>Back to workspace</Text>
          </Pressable>
        </View>
      </ScreenShell>
    );
  }

  return (
    <ScreenShell
      backHref="/processing"
      subtitle="Result cards now lead into a dedicated detail view instead of rendering every metric inline in the pipeline screen."
      title="Results"
    >
      <View style={styles.summaryRow}>
        <View style={styles.summaryChip}>
          <Text style={styles.summaryValue}>{orderedResults.length}</Text>
          <Text style={styles.summaryLabel}>photos</Text>
        </View>
        <View style={styles.summaryChip}>
          <Text style={styles.summaryValue}>{faceCount}</Text>
          <Text style={styles.summaryLabel}>faces</Text>
        </View>
        <View style={styles.summaryChip}>
          <Text style={styles.summaryValue}>{issueCount}</Text>
          <Text style={styles.summaryLabel}>issues</Text>
        </View>
      </View>
      <View style={styles.list}>
        {orderedResults.map((result) => (
          <ResultImageCard
            key={result.assetId}
            onPress={() => router.push({ pathname: "/result/[assetId]", params: { assetId: result.assetId } })}
            result={result}
          />
        ))}
      </View>
    </ScreenShell>
  );
}

const styles = StyleSheet.create({
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
  list: {
    gap: 16,
  },
  primaryButton: {
    backgroundColor: workflowTheme.accent,
    borderRadius: 18,
    paddingHorizontal: 18,
    paddingVertical: 16,
  },
  primaryLabel: {
    color: "#FFF",
    fontSize: 16,
    fontWeight: "700",
    textAlign: "center",
  },
  summaryChip: {
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 20,
    borderWidth: 1,
    flexBasis: 0,
    flexGrow: 1,
    minWidth: 96,
    padding: 16,
  },
  summaryLabel: {
    color: workflowTheme.inkMuted,
    fontSize: 12,
    marginTop: 4,
    textTransform: "uppercase",
  },
  summaryRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginBottom: 18,
  },
  summaryValue: {
    color: workflowTheme.ink,
    fontSize: 22,
    fontWeight: "700",
  },
});
