import { StyleSheet, Text, View } from "react-native";

import { workflowTheme } from "@/components/workflow/theme";
import type { AnalysisStage } from "@/types/workflow";

type ProcessingStageBadgeProps = {
  label: string;
  stage: AnalysisStage;
};

function stageStyles(stage: AnalysisStage) {
  if (stage === "completed") {
    return { backgroundColor: workflowTheme.successMuted, color: workflowTheme.success };
  }

  if (stage === "failed" || stage === "cancelled") {
    return { backgroundColor: workflowTheme.dangerMuted, color: workflowTheme.danger };
  }

  if (stage === "queued") {
    return { backgroundColor: workflowTheme.panelAlt, color: workflowTheme.ink };
  }

  return { backgroundColor: workflowTheme.warningMuted, color: workflowTheme.warning };
}

export function ProcessingStageBadge({ label, stage }: ProcessingStageBadgeProps) {
  const palette = stageStyles(stage);

  return (
    <View style={[styles.badge, { backgroundColor: palette.backgroundColor }]}>
      <Text style={[styles.label, { color: palette.color }]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    alignSelf: "flex-start",
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 7,
  },
  label: {
    fontSize: 12,
    fontWeight: "700",
  },
});
