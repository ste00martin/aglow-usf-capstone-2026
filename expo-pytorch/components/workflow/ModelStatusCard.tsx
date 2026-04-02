import { StyleSheet, Text, View, useWindowDimensions } from "react-native";

import { workflowTheme } from "@/components/workflow/theme";
import type { ModelStatus } from "@/types/workflow";

type ModelStatusCardProps = {
  modelStatuses: ModelStatus[];
  modelsReady: boolean;
  overallModelProgress: number;
};

export function ModelStatusCard({
  modelStatuses,
  modelsReady,
  overallModelProgress,
}: ModelStatusCardProps) {
  const { width } = useWindowDimensions();
  const isCompact = width < 420;

  return (
    <View style={styles.card}>
      <View style={[styles.header, isCompact ? styles.headerCompact : undefined]}>
        <View style={styles.headerCopy}>
          <Text style={styles.title}>Creator stack</Text>
          <Text style={styles.subtitle}>
            {modelsReady
              ? "Face scan, safety checks, and labels are ready to roll."
              : `Warming up the stack ${Math.round(overallModelProgress * 100)}%`}
          </Text>
        </View>
        <View style={[styles.statusPill, modelsReady ? styles.readyPill : styles.loadingPill]}>
          <Text style={[styles.statusLabel, modelsReady ? styles.readyLabel : styles.loadingLabel]}>
            {modelsReady ? "Live" : "Syncing"}
          </Text>
        </View>
      </View>
      <View style={styles.track}>
        <View style={[styles.fill, { width: `${Math.round(overallModelProgress * 100)}%` }]} />
      </View>
      <View style={styles.list}>
        {modelStatuses.map((status) => (
          <View key={status.key} style={styles.row}>
            <View style={[styles.dot, status.isReady ? styles.dotReady : styles.dotLoading]} />
            <View style={styles.rowCopy}>
              <Text style={styles.modelName}>{status.label}</Text>
              <Text style={styles.modelMeta}>
                {status.error
                  ? status.error
                  : status.isReady
                    ? "Live"
                    : `Syncing ${Math.round(status.downloadProgress * 100)}%`}
              </Text>
            </View>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 28,
    borderWidth: 1,
    gap: 14,
    padding: 18,
    shadowColor: workflowTheme.accent,
    shadowOpacity: 0.08,
    shadowRadius: 16,
    shadowOffset: { height: 8, width: 0 },
  },
  dot: {
    borderRadius: 999,
    height: 10,
    marginTop: 5,
    width: 10,
  },
  dotLoading: {
    backgroundColor: workflowTheme.warning,
  },
  dotReady: {
    backgroundColor: workflowTheme.success,
  },
  fill: {
    backgroundColor: workflowTheme.accentAlt,
    borderRadius: 999,
    height: "100%",
  },
  header: {
    alignItems: "flex-start",
    flexDirection: "row",
    gap: 12,
    justifyContent: "space-between",
  },
  headerCompact: {
    flexDirection: "column",
  },
  headerCopy: {
    flex: 1,
    minWidth: 0,
  },
  list: {
    gap: 12,
  },
  loadingLabel: {
    color: workflowTheme.accentAlt,
  },
  loadingPill: {
    backgroundColor: workflowTheme.panelAlt,
  },
  modelMeta: {
    color: workflowTheme.inkMuted,
    fontSize: 13,
  },
  modelName: {
    color: workflowTheme.ink,
    fontSize: 15,
    fontWeight: "600",
  },
  readyLabel: {
    color: workflowTheme.accent,
  },
  readyPill: {
    backgroundColor: workflowTheme.accentMuted,
  },
  row: {
    flexDirection: "row",
    gap: 10,
  },
  rowCopy: {
    flex: 1,
    gap: 3,
  },
  statusLabel: {
    fontSize: 12,
    fontWeight: "700",
    textTransform: "uppercase",
  },
  statusPill: {
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  subtitle: {
    color: workflowTheme.inkMuted,
    fontSize: 14,
    marginTop: 4,
  },
  title: {
    color: workflowTheme.ink,
    fontSize: 19,
    fontWeight: "800",
  },
  track: {
    backgroundColor: workflowTheme.accentAltMuted,
    borderRadius: 999,
    height: 10,
    overflow: "hidden",
  },
});
