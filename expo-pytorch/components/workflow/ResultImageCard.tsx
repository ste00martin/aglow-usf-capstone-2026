import { Pressable, Image, StyleSheet, Text, View } from "react-native";

import { FaceOverlay } from "@/components/workflow/FaceOverlay";
import { ResultChips } from "@/components/workflow/ResultChips";
import { workflowTheme } from "@/components/workflow/theme";
import type { ImageAnalysisResult } from "@/types/workflow";

type ResultImageCardProps = {
  onPress: () => void;
  result: ImageAnalysisResult;
};

export function ResultImageCard({ onPress, result }: ResultImageCardProps) {
  return (
    <Pressable onPress={onPress} style={styles.card}>
      <View style={styles.imageFrame}>
        <Image source={{ uri: result.uri }} style={styles.image} />
        <FaceOverlay faces={result.faces} />
      </View>
      <View style={styles.copyWrap}>
        <Text style={styles.title}>{result.error ? "Partial result" : "Ready to inspect"}</Text>
        <Text style={styles.body}>
          {result.error
            ? result.error
            : `${result.faces.length} detected face${result.faces.length === 1 ? "" : "s"} with on-device safety tags.`}
        </Text>
        <ResultChips result={result} />
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  body: {
    color: workflowTheme.inkMuted,
    fontSize: 14,
    lineHeight: 20,
  },
  card: {
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 24,
    borderWidth: 1,
    gap: 14,
    overflow: "hidden",
    padding: 14,
  },
  copyWrap: {
    gap: 8,
  },
  image: {
    height: "100%",
    width: "100%",
  },
  imageFrame: {
    borderRadius: 18,
    height: 240,
    overflow: "hidden",
  },
  title: {
    color: workflowTheme.ink,
    fontSize: 18,
    fontWeight: "700",
  },
});
