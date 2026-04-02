import { StyleSheet, View } from "react-native";

import { workflowTheme } from "@/components/workflow/theme";
import type { FaceResult } from "@/types/workflow";

type FaceOverlayProps = {
  faces: FaceResult[];
};

export function FaceOverlay({ faces }: FaceOverlayProps) {
  return (
    <View pointerEvents="none" style={styles.overlay}>
      {faces.map((face, index) => (
        <View
          key={`${face.bbox.xmin}-${face.bbox.ymin}-${index}`}
          style={[
            styles.box,
            {
              height: `${(face.bbox.ymax - face.bbox.ymin) * 100}%`,
              left: `${face.bbox.xmin * 100}%`,
              top: `${face.bbox.ymin * 100}%`,
              width: `${(face.bbox.xmax - face.bbox.xmin) * 100}%`,
            },
          ]}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  box: {
    backgroundColor: "rgba(76,141,127,0.12)",
    borderColor: workflowTheme.accent,
    borderRadius: 12,
    borderWidth: 2,
    position: "absolute",
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
  },
});
