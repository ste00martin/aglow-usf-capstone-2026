import { Ionicons } from "@expo/vector-icons";
import { StyleSheet, Text, View } from "react-native";

import { workflowTheme } from "@/components/workflow/theme";

const PILLS = [
  { icon: "sparkles-outline", label: "Creator-first" },
  { icon: "cloud-offline-outline", label: "No uploads" },
  { icon: "phone-portrait-outline", label: "Phone-fast" },
];

export function TrustPillRow() {
  return (
    <View style={styles.row}>
      {PILLS.map((pill) => (
        <View key={pill.label} style={styles.pill}>
          <Ionicons color={workflowTheme.accent} name={pill.icon as never} size={16} />
          <Text style={styles.label}>{pill.label}</Text>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  label: {
    color: workflowTheme.ink,
    fontSize: 13,
    fontWeight: "600",
  },
  pill: {
    alignItems: "center",
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 999,
    borderWidth: 1,
    flexDirection: "row",
    gap: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    shadowColor: workflowTheme.accent,
    shadowOpacity: 0.06,
    shadowRadius: 10,
    shadowOffset: { height: 4, width: 0 },
  },
  row: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 16,
  },
});
