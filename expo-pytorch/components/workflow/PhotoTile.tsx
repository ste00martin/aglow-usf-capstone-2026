import { Ionicons } from "@expo/vector-icons";
import { Image, Pressable, StyleSheet, Text, View } from "react-native";
import type * as MediaLibrary from "expo-media-library";

import { workflowTheme } from "@/components/workflow/theme";

type PhotoTileProps = {
  asset: MediaLibrary.Asset;
  isSelected: boolean;
  onPress: () => void;
};

export function PhotoTile({ asset, isSelected, onPress }: PhotoTileProps) {
  return (
    <Pressable onPress={onPress} style={[styles.tile, isSelected ? styles.tileSelected : undefined]}>
      <Image source={{ uri: asset.uri }} style={styles.image} />
      {isSelected ? <View style={styles.selectedOverlay} /> : null}
      <View style={[styles.badge, isSelected ? styles.badgeSelected : undefined]}>
        {isSelected ? (
          <Ionicons color="#FFF" name="checkmark" size={16} />
        ) : (
          <Text style={styles.badgeLabel}>+</Text>
        )}
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  badge: {
    alignItems: "center",
    backgroundColor: "rgba(31,42,36,0.72)",
    borderRadius: 999,
    height: 28,
    justifyContent: "center",
    position: "absolute",
    right: 10,
    top: 10,
    width: 28,
  },
  badgeLabel: {
    color: "#FFF",
    fontSize: 16,
    fontWeight: "600",
  },
  badgeSelected: {
    backgroundColor: workflowTheme.accent,
  },
  image: {
    borderRadius: 18,
    flex: 1,
  },
  selectedOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "rgba(76,141,127,0.24)",
    borderRadius: 18,
  },
  tile: {
    aspectRatio: 1,
    borderRadius: 18,
    overflow: "hidden",
    width: "31.4%",
  },
  tileSelected: {
    borderColor: workflowTheme.accent,
    borderWidth: 2,
  },
});
