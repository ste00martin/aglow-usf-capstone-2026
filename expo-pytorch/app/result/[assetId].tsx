import { router, useLocalSearchParams } from "expo-router";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { useWorkflow } from "@/AlbumContext";
import { ResultDetailSheet } from "@/components/workflow/ResultDetailSheet";
import { ScreenShell } from "@/components/workflow/ScreenShell";
import { workflowTheme } from "@/components/workflow/theme";

export default function ResultDetailScreen() {
  const { assetId } = useLocalSearchParams<{ assetId: string }>();
  const { resultsByAsset } = useWorkflow();

  const result = assetId ? resultsByAsset[assetId] : undefined;

  if (!result) {
    return (
      <ScreenShell
        backHref="/results"
        subtitle="The requested result is not available in the current workflow state."
        title="Result detail"
      >
        <View style={styles.emptyCard}>
          <Text style={styles.emptyTitle}>Result not found.</Text>
          <Pressable style={styles.primaryButton} onPress={() => router.back()}>
            <Text style={styles.primaryLabel}>Back</Text>
          </Pressable>
        </View>
      </ScreenShell>
    );
  }

  return (
    <ScreenShell
      backHref="/results"
      subtitle="Detailed face and safety metadata now live in a dedicated route."
      title="Result detail"
    >
      <ResultDetailSheet result={result} />
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
});
