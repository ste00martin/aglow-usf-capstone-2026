import { router } from "expo-router";
import { Pressable, StyleSheet, Text, View, useWindowDimensions } from "react-native";

import { useWorkflow } from "@/AlbumContext";
import { ModelStatusCard } from "@/components/workflow/ModelStatusCard";
import { PermissionHero } from "@/components/workflow/PermissionHero";
import { PhotoGrid } from "@/components/workflow/PhotoGrid";
import { ScreenShell } from "@/components/workflow/ScreenShell";
import { SelectionActionBar } from "@/components/workflow/SelectionActionBar";
import { TrustPillRow } from "@/components/workflow/TrustPillRow";
import { workflowTheme } from "@/components/workflow/theme";

type SuggestionList = {
  eyebrow: string;
  items: string[];
  title: string;
  tone: "cool" | "warm";
};

const FORMAT_SUGGESTIONS = ["Photo dump", "GRWM", "Outfit recap", "Campus candid", "Night-out carousel"];

const LOCKED_SUGGESTIONS: SuggestionList[] = [
  {
    eyebrow: "First moves",
    items: [
      "Unlock the camera roll so you can browse real posts-in-progress.",
      "Pick 6 to 12 photos for a strong social batch.",
      "Start with a mix of selfies, candids, and group shots.",
    ],
    title: "How to open the feed lab",
    tone: "warm",
  },
  {
    eyebrow: "Best mix",
    items: [
      "Lead with your brightest close-up.",
      "Keep one wide shot for scene-setting.",
      "Save the most surprising frame for the middle of the carousel.",
    ],
    title: "What usually feels post-worthy",
    tone: "cool",
  },
];

const READY_SUGGESTIONS: SuggestionList[] = [
  {
    eyebrow: "Batch ideas",
    items: [
      "Golden-hour carousel with one hero image up front.",
      "Study-break photo dump with one candid every other slide.",
      "Outfit-check recap with a detail shot near the end.",
    ],
    title: "Try one of these post formats",
    tone: "warm",
  },
  {
    eyebrow: "Caption prompts",
    items: ["Soft-launch energy.", "POV: the camera roll finally delivered.", "Main-character break between classes."],
    title: "Quick lines that fit the vibe",
    tone: "cool",
  },
];

function SuggestionListCard({ eyebrow, items, title, tone }: SuggestionList) {
  return (
    <View style={[styles.suggestionCard, tone === "cool" ? styles.suggestionCardCool : styles.suggestionCardWarm]}>
      <Text style={styles.suggestionEyebrow}>{eyebrow}</Text>
      <Text style={styles.suggestionTitle}>{title}</Text>
      <View style={styles.suggestionList}>
        {items.map((item) => (
          <View key={item} style={styles.suggestionItem}>
            <View style={styles.suggestionDot} />
            <Text style={styles.suggestionText}>{item}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

export default function WorkspaceScreen() {
  const { width } = useWindowDimensions();
  const isCompact = width < 420;
  const {
    clearSelection,
    isLibraryLoading,
    libraryAssets,
    libraryError,
    modelStatuses,
    modelsReady,
    openSystemSettings,
    overallModelProgress,
    permissionState,
    presentPermissionsPicker,
    refreshLibrary,
    requestLibraryPermission,
    selectedAssetIds,
    toggleAssetSelection,
  } = useWorkflow();

  const suggestionLists = permissionState.isGranted ? READY_SUGGESTIONS : LOCKED_SUGGESTIONS;
  const pulseCards = [
    { label: "Selected", value: String(selectedAssetIds.length) },
    { label: "Library", value: permissionState.isGranted ? String(libraryAssets.length) : "Locked" },
    { label: "Stack", value: modelsReady ? "Live" : `${Math.round(overallModelProgress * 100)}%` },
  ];
  const heroTitle = !permissionState.isGranted
    ? "Start with a camera roll unlock."
    : selectedAssetIds.length
      ? `${selectedAssetIds.length} shots are lined up for your next drop.`
      : "Build a playful batch before you hit review.";
  const heroBody = !permissionState.isGranted
    ? "Give the app access to your photos, then use the suggestion lists below to shape a more social-looking post set."
    : selectedAssetIds.length
      ? "You already have a batch going. Tighten the mood with the suggestion lists, then send it to review."
      : "Pick a few standout frames, mix in one candid, and use the suggestion lists below to shape the vibe.";

  return (
    <View style={styles.root}>
      <ScreenShell
        contentContainerStyle={{
          paddingBottom: selectedAssetIds.length ? 148 : 32,
        }}
        subtitle="Pick a photo drop, browse creator suggestion lists, and queue a private on-device scan before you post."
        title="Feed Lab"
      >
        <View style={styles.heroCard}>
          <View style={styles.heroGlow} />
          <View style={[styles.heroHeader, isCompact ? styles.heroHeaderCompact : undefined]}>
            <View style={[styles.heroCopy, isCompact ? styles.heroCopyCompact : undefined]}>
              <Text style={styles.heroEyebrow}>{"Today's vibe"}</Text>
              <Text style={styles.heroTitle}>{heroTitle}</Text>
              <Text style={styles.heroBody}>{heroBody}</Text>
            </View>
            <View
              style={[
                styles.heroBadge,
                modelsReady ? styles.heroBadgeReady : styles.heroBadgeLoading,
                isCompact ? styles.heroBadgeCompact : undefined,
              ]}
            >
              <Text style={[styles.heroBadgeLabel, modelsReady ? styles.heroBadgeLabelReady : styles.heroBadgeLabelLoading]}>
                {modelsReady ? "Ready to post" : "Warming up"}
              </Text>
            </View>
          </View>
          <View style={styles.pulseRow}>
            {pulseCards.map((card) => (
              <View key={card.label} style={styles.pulseCard}>
                <Text style={styles.pulseValue}>{card.value}</Text>
                <Text style={styles.pulseLabel}>{card.label}</Text>
              </View>
            ))}
          </View>
          <View style={styles.formatRow}>
            {FORMAT_SUGGESTIONS.map((format) => (
              <View key={format} style={styles.formatChip}>
                <Text style={styles.formatLabel}>{format}</Text>
              </View>
            ))}
          </View>
        </View>

        <PermissionHero
          assetCount={libraryAssets.length}
          isLibraryLoading={isLibraryLoading}
          onManageAccess={permissionState.isLimited ? presentPermissionsPicker : refreshLibrary}
          onOpenSettings={openSystemSettings}
          onRequestPermission={requestLibraryPermission}
          permissionState={permissionState}
        />

        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Suggestion lists</Text>
            <Text style={styles.sectionBody}>
              This screen now gives you actual social-style prompts instead of just operational status cards.
            </Text>
          </View>
          <View style={styles.suggestionGrid}>
            {suggestionLists.map((list) => (
              <SuggestionListCard key={list.title} {...list} />
            ))}
          </View>
        </View>

        <TrustPillRow />

        <View style={styles.section}>
          <ModelStatusCard
            modelStatuses={modelStatuses}
            modelsReady={modelsReady}
            overallModelProgress={overallModelProgress}
          />
        </View>

        {libraryError ? (
          <View style={styles.errorCard}>
            <Text style={styles.errorTitle}>Library refresh failed</Text>
            <Text style={styles.errorBody}>{libraryError}</Text>
          </View>
        ) : null}

        <View style={styles.libraryCard}>
          <View style={[styles.gridHeader, isCompact ? styles.gridHeaderCompact : undefined]}>
            <View style={styles.gridCopy}>
              <Text style={styles.sectionTitle}>Build your next photo drop</Text>
              <Text style={styles.sectionBody}>
                Pick the frames you want to analyze. A mix of close-ups, group shots, and candid moments feels more like a real feed.
              </Text>
            </View>
            <Pressable
              onPress={permissionState.isGranted ? refreshLibrary : requestLibraryPermission}
              style={[styles.refreshButton, isCompact ? styles.refreshButtonCompact : undefined]}
            >
              <Text style={styles.refreshLabel}>{permissionState.isGranted ? "Refresh" : "Allow access"}</Text>
            </Pressable>
          </View>
          <PhotoGrid
            assets={permissionState.isGranted ? libraryAssets : []}
            selectedAssetIds={selectedAssetIds}
            onToggle={toggleAssetSelection}
          />
        </View>
      </ScreenShell>
      {selectedAssetIds.length ? (
        <SelectionActionBar
          count={selectedAssetIds.length}
          onClear={clearSelection}
          onReview={() => router.push("/review")}
        />
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  errorBody: {
    color: workflowTheme.danger,
    fontSize: 14,
    lineHeight: 20,
  },
  errorCard: {
    backgroundColor: workflowTheme.dangerMuted,
    borderRadius: 24,
    gap: 4,
    marginTop: 18,
    padding: 16,
  },
  errorTitle: {
    color: workflowTheme.danger,
    fontSize: 16,
    fontWeight: "700",
  },
  formatChip: {
    backgroundColor: "rgba(255,255,255,0.74)",
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  formatLabel: {
    color: workflowTheme.ink,
    fontSize: 13,
    fontWeight: "700",
  },
  formatRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 18,
  },
  gridCopy: {
    flex: 1,
    gap: 4,
  },
  gridHeader: {
    alignItems: "flex-start",
    flexDirection: "row",
    gap: 14,
    justifyContent: "space-between",
    marginBottom: 16,
  },
  gridHeaderCompact: {
    flexDirection: "column",
  },
  heroBadge: {
    borderRadius: 999,
    paddingHorizontal: 12,
    paddingVertical: 10,
  },
  heroBadgeCompact: {
    alignSelf: "flex-start",
  },
  heroBadgeLabel: {
    fontSize: 12,
    fontWeight: "800",
    textTransform: "uppercase",
  },
  heroBadgeLabelLoading: {
    color: workflowTheme.accentAlt,
  },
  heroBadgeLabelReady: {
    color: workflowTheme.accent,
  },
  heroBadgeLoading: {
    backgroundColor: workflowTheme.accentAltMuted,
  },
  heroBadgeReady: {
    backgroundColor: workflowTheme.accentMuted,
  },
  heroBody: {
    color: workflowTheme.inkMuted,
    fontSize: 15,
    lineHeight: 22,
  },
  heroCard: {
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 30,
    borderWidth: 1,
    marginBottom: 18,
    overflow: "hidden",
    padding: 20,
    position: "relative",
    shadowColor: workflowTheme.accent,
    shadowOpacity: 0.12,
    shadowRadius: 22,
    shadowOffset: { height: 12, width: 0 },
  },
  heroCopy: {
    flex: 1,
    gap: 8,
    paddingRight: 12,
  },
  heroCopyCompact: {
    paddingRight: 0,
  },
  heroEyebrow: {
    color: workflowTheme.accent,
    fontSize: 12,
    fontWeight: "800",
    letterSpacing: 1.2,
    textTransform: "uppercase",
  },
  heroGlow: {
    backgroundColor: workflowTheme.panelAlt,
    borderRadius: 999,
    height: 160,
    position: "absolute",
    right: -24,
    top: -32,
    width: 160,
  },
  heroHeader: {
    alignItems: "flex-start",
    flexDirection: "row",
    gap: 12,
    justifyContent: "space-between",
  },
  heroHeaderCompact: {
    flexDirection: "column",
  },
  heroTitle: {
    color: workflowTheme.ink,
    fontSize: 28,
    fontWeight: "800",
    letterSpacing: -0.8,
  },
  libraryCard: {
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 28,
    borderWidth: 1,
    marginTop: 22,
    padding: 18,
    shadowColor: workflowTheme.accent,
    shadowOpacity: 0.06,
    shadowRadius: 14,
    shadowOffset: { height: 8, width: 0 },
  },
  pulseCard: {
    backgroundColor: "rgba(255,255,255,0.88)",
    borderRadius: 20,
    flex: 1,
    minWidth: 92,
    paddingHorizontal: 14,
    paddingVertical: 14,
  },
  pulseLabel: {
    color: workflowTheme.inkMuted,
    fontSize: 12,
    marginTop: 4,
    textTransform: "uppercase",
  },
  pulseRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginTop: 18,
  },
  pulseValue: {
    color: workflowTheme.ink,
    fontSize: 22,
    fontWeight: "800",
  },
  refreshButton: {
    backgroundColor: workflowTheme.accentAltMuted,
    borderRadius: 999,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  refreshButtonCompact: {
    width: "100%",
  },
  refreshLabel: {
    color: workflowTheme.ink,
    fontSize: 14,
    fontWeight: "700",
  },
  root: {
    backgroundColor: workflowTheme.background,
    flex: 1,
  },
  section: {
    marginTop: 22,
  },
  sectionBody: {
    color: workflowTheme.inkMuted,
    fontSize: 14,
    lineHeight: 20,
    marginTop: 4,
  },
  sectionHeader: {
    marginBottom: 14,
  },
  sectionTitle: {
    color: workflowTheme.ink,
    fontSize: 20,
    fontWeight: "800",
    letterSpacing: -0.4,
  },
  suggestionCard: {
    borderRadius: 26,
    flex: 1,
    minWidth: 160,
    padding: 18,
  },
  suggestionCardCool: {
    backgroundColor: workflowTheme.accentAltMuted,
  },
  suggestionCardWarm: {
    backgroundColor: workflowTheme.accentMuted,
  },
  suggestionDot: {
    backgroundColor: workflowTheme.ink,
    borderRadius: 999,
    height: 6,
    marginTop: 7,
    width: 6,
  },
  suggestionEyebrow: {
    color: workflowTheme.inkMuted,
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 1,
    textTransform: "uppercase",
  },
  suggestionGrid: {
    gap: 12,
  },
  suggestionItem: {
    flexDirection: "row",
    gap: 10,
  },
  suggestionList: {
    gap: 10,
    marginTop: 12,
  },
  suggestionText: {
    color: workflowTheme.ink,
    flex: 1,
    fontSize: 14,
    lineHeight: 20,
  },
  suggestionTitle: {
    color: workflowTheme.ink,
    fontSize: 20,
    fontWeight: "800",
    letterSpacing: -0.5,
    marginTop: 6,
  },
});
