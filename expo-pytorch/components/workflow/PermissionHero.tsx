import { Ionicons } from "@expo/vector-icons";
import { Pressable, StyleSheet, Text, View, useWindowDimensions } from "react-native";

import { workflowTheme } from "@/components/workflow/theme";
import type { PermissionState } from "@/types/workflow";

type PermissionHeroProps = {
  assetCount: number;
  isLibraryLoading: boolean;
  permissionState: PermissionState;
  onManageAccess: () => void;
  onOpenSettings: () => void;
  onRequestPermission: () => void;
};

function getCopy(permissionState: PermissionState): {
  title: string;
  subtitle: string;
  primaryLabel: string;
} {
  if (!permissionState.isGranted) {
    return {
      primaryLabel: "Unlock camera roll",
      subtitle:
        "Your photos stay on this phone. Pull in a batch, try a few post ideas, and keep the whole review flow private.",
      title: "Give Aglow access and start building your next drop.",
    };
  }

  if (permissionState.isLimited) {
    return {
      primaryLabel: "Pick more photos",
      subtitle:
        "You are browsing a smaller slice of the camera roll. Add more shots for better mix-and-match social post ideas.",
      title: "This batch is cute, but your feed could use more range.",
    };
  }

  return {
    primaryLabel: "Refresh the mix",
    subtitle:
      "Choose a batch, skim the suggestion lists, and queue a private creator scan before you post anything.",
    title: "Your feed lab is live.",
  };
}

export function PermissionHero({
  assetCount,
  isLibraryLoading,
  onManageAccess,
  onOpenSettings,
  onRequestPermission,
  permissionState,
}: PermissionHeroProps) {
  const { width } = useWindowDimensions();
  const isCompact = width < 420;
  const copy = getCopy(permissionState);
  const primaryAction = permissionState.isGranted ? onManageAccess : onRequestPermission;

  return (
    <View style={styles.card}>
      <View style={styles.spotlight} />
      <View style={[styles.header, isCompact ? styles.headerCompact : undefined]}>
        <View style={styles.iconWrap}>
          <Ionicons name="sparkles-outline" color={workflowTheme.accent} size={24} />
        </View>
        <View style={styles.copyWrap}>
          <Text style={styles.title}>{copy.title}</Text>
          <Text style={styles.subtitle}>{copy.subtitle}</Text>
        </View>
      </View>
      <View style={styles.metaRow}>
        <View style={styles.metaPill}>
          <Text style={styles.metaLabel}>
            {permissionState.isGranted ? `${assetCount} photos in rotation` : "Camera roll locked"}
          </Text>
        </View>
        <View style={styles.metaPill}>
          <Text style={styles.metaLabel}>{permissionState.isGranted ? "Private by default" : "Stays on-device"}</Text>
        </View>
        {isLibraryLoading ? (
          <View style={[styles.metaPill, styles.loadingPill]}>
            <Text style={styles.metaLabel}>Refreshing photos...</Text>
          </View>
        ) : null}
      </View>
      <View style={[styles.actions, isCompact ? styles.actionsCompact : undefined]}>
        <Pressable style={[styles.primaryButton, isCompact ? styles.buttonCompact : undefined]} onPress={primaryAction}>
          <Text style={styles.primaryLabel}>{copy.primaryLabel}</Text>
        </Pressable>
        <Pressable
          style={[styles.secondaryButton, isCompact ? styles.buttonCompact : undefined]}
          onPress={onOpenSettings}
        >
          <Text style={styles.secondaryLabel}>Open Settings</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  actions: {
    flexDirection: "row",
    gap: 10,
    marginTop: 18,
  },
  actionsCompact: {
    flexDirection: "column",
  },
  buttonCompact: {
    width: "100%",
  },
  card: {
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 30,
    borderWidth: 1,
    overflow: "hidden",
    padding: 20,
    shadowColor: workflowTheme.accent,
    shadowOpacity: 0.14,
    shadowRadius: 18,
    shadowOffset: { height: 6, width: 0 },
  },
  copyWrap: {
    flex: 1,
    gap: 6,
  },
  header: {
    flexDirection: "row",
    gap: 14,
  },
  headerCompact: {
    alignItems: "flex-start",
    flexDirection: "column",
  },
  iconWrap: {
    alignItems: "center",
    backgroundColor: workflowTheme.accentMuted,
    borderRadius: 18,
    height: 52,
    justifyContent: "center",
    width: 52,
  },
  loadingPill: {
    backgroundColor: workflowTheme.accentAltMuted,
  },
  metaLabel: {
    color: workflowTheme.ink,
    fontSize: 13,
    fontWeight: "600",
  },
  metaPill: {
    backgroundColor: workflowTheme.panelAlt,
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  metaRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 16,
  },
  primaryButton: {
    backgroundColor: workflowTheme.accent,
    borderRadius: 16,
    flex: 1,
    paddingHorizontal: 14,
    paddingVertical: 14,
  },
  primaryLabel: {
    color: "#FFF",
    fontSize: 15,
    fontWeight: "700",
    textAlign: "center",
  },
  secondaryButton: {
    backgroundColor: workflowTheme.accentAltMuted,
    borderRadius: 16,
    minWidth: 136,
    paddingHorizontal: 14,
    paddingVertical: 14,
  },
  secondaryLabel: {
    color: workflowTheme.ink,
    fontSize: 15,
    fontWeight: "600",
  },
  subtitle: {
    color: workflowTheme.inkMuted,
    fontSize: 15,
    lineHeight: 22,
  },
  spotlight: {
    backgroundColor: workflowTheme.panelAlt,
    borderRadius: 999,
    height: 120,
    position: "absolute",
    right: -30,
    top: -20,
    width: 120,
  },
  title: {
    color: workflowTheme.ink,
    fontSize: 24,
    fontWeight: "800",
    letterSpacing: -0.6,
  },
});
