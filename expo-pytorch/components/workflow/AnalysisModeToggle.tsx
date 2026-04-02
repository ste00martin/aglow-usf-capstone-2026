import { Pressable, StyleSheet, Text, View } from "react-native";

import { workflowTheme } from "@/components/workflow/theme";
import { ANALYSIS_MODE_COPY, type AnalysisMode } from "@/types/workflow";

type AnalysisModeToggleProps = {
  mode: AnalysisMode;
  onChange: (mode: AnalysisMode) => void;
};

const MODES: AnalysisMode[] = ["quick", "full"];

export function AnalysisModeToggle({ mode, onChange }: AnalysisModeToggleProps) {
  return (
    <View style={styles.card}>
      <Text style={styles.title}>Analysis mode</Text>
      <View style={styles.list}>
        {MODES.map((candidate) => {
          const isActive = candidate === mode;

          return (
            <Pressable
              key={candidate}
              onPress={() => onChange(candidate)}
              style={[styles.option, isActive ? styles.optionActive : undefined]}
            >
              <Text style={[styles.optionTitle, isActive ? styles.optionTitleActive : undefined]}>
                {ANALYSIS_MODE_COPY[candidate].title}
              </Text>
              <Text style={[styles.optionBody, isActive ? styles.optionBodyActive : undefined]}>
                {ANALYSIS_MODE_COPY[candidate].subtitle}
              </Text>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 24,
    borderWidth: 1,
    gap: 14,
    padding: 18,
  },
  list: {
    gap: 10,
  },
  option: {
    backgroundColor: workflowTheme.panelAlt,
    borderRadius: 18,
    gap: 6,
    padding: 14,
  },
  optionActive: {
    backgroundColor: workflowTheme.accent,
  },
  optionBody: {
    color: workflowTheme.inkMuted,
    fontSize: 13,
    lineHeight: 19,
  },
  optionBodyActive: {
    color: "rgba(255,255,255,0.84)",
  },
  optionTitle: {
    color: workflowTheme.ink,
    fontSize: 16,
    fontWeight: "700",
  },
  optionTitleActive: {
    color: "#FFF",
  },
  title: {
    color: workflowTheme.ink,
    fontSize: 19,
    fontWeight: "700",
  },
});
