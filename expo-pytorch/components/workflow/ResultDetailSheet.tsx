import { Image, StyleSheet, Text, View, useWindowDimensions } from "react-native";

import { FaceOverlay } from "@/components/workflow/FaceOverlay";
import { ResultChips } from "@/components/workflow/ResultChips";
import { workflowTheme } from "@/components/workflow/theme";
import type { ImageAnalysisResult } from "@/types/workflow";

type ResultDetailSheetProps = {
  result: ImageAnalysisResult;
};

export function ResultDetailSheet({ result }: ResultDetailSheetProps) {
  const { width } = useWindowDimensions();
  const heroHeight = Math.min(360, Math.max(240, width - 40));

  return (
    <View style={styles.stack}>
      <View style={[styles.hero, { height: heroHeight }]}>
        <Image source={{ uri: result.uri }} style={styles.heroImage} />
        <FaceOverlay faces={result.faces} />
      </View>
      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Summary</Text>
        <ResultChips result={result} />
      </View>
      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Detected faces</Text>
        {result.faces.length ? (
          result.faces.map((face, index) => (
            <View key={`${face.bbox.xmin}-${face.bbox.ymin}-${index}`} style={styles.faceRow}>
              <Text style={styles.faceTitle}>Face {index + 1}</Text>
              <Text style={styles.faceMeta}>
                Age {face.age.label} ({(face.age.score * 100).toFixed(1)}%)
              </Text>
              <Text style={styles.faceMeta}>
                Gender {face.gender.label} ({(face.gender.score * 100).toFixed(1)}%)
              </Text>
            </View>
          ))
        ) : (
          <Text style={styles.emptyText}>No faces were detected in this photo.</Text>
        )}
      </View>
      <View style={styles.card}>
        <Text style={styles.sectionTitle}>Safety scores</Text>
        {result.nsfw.length ? (
          result.nsfw.slice(0, 4).map((item) => (
            <View key={item.label} style={styles.scoreRow}>
              <Text style={styles.scoreLabel}>{item.label}</Text>
              <Text style={styles.scoreValue}>{(item.score * 100).toFixed(1)}%</Text>
            </View>
          ))
        ) : (
          <Text style={styles.emptyText}>No safety scores were returned.</Text>
        )}
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
    gap: 12,
    padding: 18,
  },
  emptyText: {
    color: workflowTheme.inkMuted,
    fontSize: 14,
    lineHeight: 20,
  },
  faceMeta: {
    color: workflowTheme.inkMuted,
    fontSize: 14,
    lineHeight: 20,
  },
  faceRow: {
    borderTopColor: workflowTheme.border,
    borderTopWidth: 1,
    gap: 4,
    paddingTop: 12,
  },
  faceTitle: {
    color: workflowTheme.ink,
    fontSize: 16,
    fontWeight: "700",
  },
  hero: {
    borderRadius: 28,
    overflow: "hidden",
  },
  heroImage: {
    height: "100%",
    width: "100%",
  },
  scoreLabel: {
    color: workflowTheme.ink,
    flex: 1,
    fontSize: 15,
    fontWeight: "600",
    paddingRight: 12,
  },
  scoreRow: {
    alignItems: "flex-start",
    borderTopColor: workflowTheme.border,
    borderTopWidth: 1,
    flexDirection: "row",
    gap: 12,
    justifyContent: "space-between",
    paddingTop: 12,
  },
  scoreValue: {
    color: workflowTheme.inkMuted,
    fontSize: 15,
  },
  sectionTitle: {
    color: workflowTheme.ink,
    fontSize: 19,
    fontWeight: "700",
  },
  stack: {
    gap: 16,
  },
});
