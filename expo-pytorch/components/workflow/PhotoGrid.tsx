import { StyleSheet, Text, View } from "react-native";
import type * as MediaLibrary from "expo-media-library";

import { PhotoTile } from "@/components/workflow/PhotoTile";
import { workflowTheme } from "@/components/workflow/theme";

type PhotoGridProps = {
  assets: MediaLibrary.Asset[];
  selectedAssetIds: string[];
  onToggle: (assetId: string) => void;
};

export function PhotoGrid({ assets, onToggle, selectedAssetIds }: PhotoGridProps) {
  if (!assets.length) {
    return (
      <View style={styles.emptyCard}>
        <Text style={styles.emptyTitle}>Your next drop has not loaded yet.</Text>
        <Text style={styles.emptySubtitle}>
          Unlock the camera roll or refresh this feed lab to pull in fresh shots.
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.grid}>
      {assets.map((asset) => (
        <PhotoTile
          key={asset.id}
          asset={asset}
          isSelected={selectedAssetIds.includes(asset.id)}
          onPress={() => onToggle(asset.id)}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  emptyCard: {
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 28,
    borderWidth: 1,
    gap: 6,
    padding: 20,
  },
  emptySubtitle: {
    color: workflowTheme.inkMuted,
    fontSize: 14,
    lineHeight: 20,
  },
  emptyTitle: {
    color: workflowTheme.ink,
    fontSize: 17,
    fontWeight: "700",
  },
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
});
