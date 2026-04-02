import { Pressable, StyleSheet, Text, View, useWindowDimensions } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { workflowTheme } from "@/components/workflow/theme";

type SelectionActionBarProps = {
  count: number;
  onClear: () => void;
  onReview: () => void;
};

export function SelectionActionBar({ count, onClear, onReview }: SelectionActionBarProps) {
  const insets = useSafeAreaInsets();
  const { width } = useWindowDimensions();
  const isCompact = width < 420;

  return (
    <View style={[styles.container, { bottom: Math.max(insets.bottom, 16) }]}>
      <View style={[styles.card, isCompact ? styles.cardCompact : undefined]}>
        <View style={styles.copyWrap}>
          <Text style={styles.countLabel}>{count} shots picked</Text>
          <Text style={styles.subLabel}>Build this drop before you run the creator scan.</Text>
        </View>
        <View style={[styles.actions, isCompact ? styles.actionsCompact : undefined]}>
          <Pressable style={[styles.clearButton, isCompact ? styles.actionButtonCompact : undefined]} onPress={onClear}>
            <Text style={styles.clearLabel}>Clear</Text>
          </Pressable>
          <Pressable
            style={[styles.reviewButton, isCompact ? styles.actionButtonCompact : undefined]}
            onPress={onReview}
          >
            <Text style={styles.reviewLabel}>Review drop</Text>
          </Pressable>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  actionButtonCompact: {
    flex: 1,
  },
  actions: {
    alignItems: "center",
    flexDirection: "row",
    flexShrink: 1,
    gap: 10,
    justifyContent: "flex-end",
  },
  actionsCompact: {
    width: "100%",
  },
  card: {
    alignItems: "center",
    backgroundColor: workflowTheme.ink,
    borderColor: "rgba(255,255,255,0.08)",
    borderRadius: 26,
    borderWidth: 1,
    flexDirection: "row",
    gap: 12,
    justifyContent: "space-between",
    paddingHorizontal: 18,
    paddingVertical: 16,
    shadowColor: workflowTheme.accent,
    shadowOpacity: 0.22,
    shadowRadius: 20,
    shadowOffset: { height: 10, width: 0 },
  },
  cardCompact: {
    alignItems: "stretch",
    flexDirection: "column",
  },
  clearButton: {
    backgroundColor: "rgba(255,255,255,0.12)",
    borderRadius: 16,
    paddingHorizontal: 14,
    paddingVertical: 12,
  },
  clearLabel: {
    color: "#FFF",
    fontSize: 14,
    fontWeight: "600",
  },
  container: {
    left: 20,
    position: "absolute",
    right: 20,
  },
  copyWrap: {
    flex: 1,
    minWidth: 0,
  },
  countLabel: {
    color: "#FFF",
    fontSize: 17,
    fontWeight: "800",
  },
  reviewButton: {
    backgroundColor: workflowTheme.accent,
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  reviewLabel: {
    color: "#FFF",
    fontSize: 14,
    fontWeight: "700",
  },
  subLabel: {
    color: "rgba(255,255,255,0.72)",
    fontSize: 13,
    lineHeight: 18,
    marginTop: 2,
  },
});
