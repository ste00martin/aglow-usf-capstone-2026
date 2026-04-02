import { StyleSheet, Text, View } from "react-native";

import { workflowTheme } from "@/components/workflow/theme";
import { formatClassificationResults } from "@/services/analysisPipeline";
import type { ImageAnalysisResult } from "@/types/workflow";

type ResultChipsProps = {
  result: ImageAnalysisResult;
};

function topSafetyLabel(result: ImageAnalysisResult): string {
  if (!result.nsfw.length) {
    return "No safety output";
  }

  return formatClassificationResults(result.nsfw, 2);
}

export function ResultChips({ result }: ResultChipsProps) {
  const chips = [
    `${result.faces.length} face${result.faces.length === 1 ? "" : "s"}`,
    topSafetyLabel(result),
  ];

  if (result.error) {
    chips.push("Needs retry");
  }

  return (
    <View style={styles.row}>
      {chips.map((chip) => (
        <View key={chip} style={styles.chip}>
          <Text style={styles.chipLabel}>{chip}</Text>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  chip: {
    backgroundColor: workflowTheme.panelAlt,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  chipLabel: {
    color: workflowTheme.ink,
    fontSize: 12,
    fontWeight: "600",
  },
  row: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
});
