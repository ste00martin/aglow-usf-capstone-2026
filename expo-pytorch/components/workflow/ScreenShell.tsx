import { Ionicons } from "@expo/vector-icons";
import type { Href } from "expo-router";
import { useRouter } from "expo-router";
import type { ReactNode } from "react";
import type { StyleProp, ViewStyle } from "react-native";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { workflowTheme } from "@/components/workflow/theme";

type ScreenShellProps = {
  backHref?: Href;
  backLabel?: string;
  title: string;
  subtitle?: string;
  children: ReactNode;
  scroll?: boolean;
  contentContainerStyle?: StyleProp<ViewStyle>;
  eyebrow?: string;
};

export function ScreenShell({
  backHref,
  backLabel = "Back",
  children,
  contentContainerStyle,
  eyebrow = "Aglow",
  scroll = true,
  subtitle,
  title,
}: ScreenShellProps) {
  const router = useRouter();

  function handleBack() {
    if (!backHref) {
      return;
    }

    if (router.canGoBack()) {
      router.back();
      return;
    }

    router.replace(backHref);
  }

  const content = (
    <View style={styles.inner}>
      <View style={styles.header}>
        {backHref ? (
          <Pressable onPress={handleBack} style={styles.backButton}>
            <Ionicons color={workflowTheme.ink} name="chevron-back" size={18} />
            <Text style={styles.backLabel}>{backLabel}</Text>
          </Pressable>
        ) : null}
        <Text style={styles.eyebrow}>{eyebrow}</Text>
        <Text style={styles.title}>{title}</Text>
        {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
      </View>
      {children}
    </View>
  );

  return (
    <SafeAreaView edges={["top", "right", "bottom", "left"]} style={styles.safeArea}>
      <View pointerEvents="none" style={styles.backdrop}>
        <View style={[styles.blob, styles.blobOne]} />
        <View style={[styles.blob, styles.blobTwo]} />
      </View>
      {scroll ? (
        <ScrollView
          contentContainerStyle={[styles.scrollContent, contentContainerStyle]}
          showsVerticalScrollIndicator={false}
        >
          {content}
        </ScrollView>
      ) : (
        <View style={[styles.scrollContent, contentContainerStyle]}>{content}</View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  backButton: {
    alignItems: "center",
    alignSelf: "flex-start",
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 999,
    borderWidth: 1,
    flexDirection: "row",
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  backLabel: {
    color: workflowTheme.ink,
    fontSize: 14,
    fontWeight: "700",
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    overflow: "hidden",
  },
  blob: {
    borderRadius: 999,
    position: "absolute",
  },
  blobOne: {
    backgroundColor: workflowTheme.accentMuted,
    height: 260,
    opacity: 0.95,
    right: -72,
    top: -44,
    width: 260,
  },
  blobTwo: {
    backgroundColor: workflowTheme.accentAltMuted,
    height: 180,
    left: -44,
    opacity: 0.9,
    top: 120,
    width: 180,
  },
  eyebrow: {
    alignSelf: "flex-start",
    backgroundColor: workflowTheme.panel,
    borderColor: workflowTheme.border,
    borderRadius: 999,
    borderWidth: 1,
    color: workflowTheme.accent,
    fontSize: 12,
    fontWeight: "700",
    letterSpacing: 1.4,
    paddingHorizontal: 12,
    paddingVertical: 8,
    textTransform: "uppercase",
  },
  header: {
    gap: 10,
    marginBottom: 24,
    paddingTop: 4,
  },
  inner: {
    flex: 1,
  },
  safeArea: {
    backgroundColor: workflowTheme.background,
    flex: 1,
    position: "relative",
  },
  scrollContent: {
    flexGrow: 1,
    paddingBottom: 28,
    paddingHorizontal: 20,
  },
  subtitle: {
    color: workflowTheme.inkMuted,
    fontSize: 15,
    lineHeight: 22,
    maxWidth: 360,
  },
  title: {
    color: workflowTheme.ink,
    fontSize: 36,
    fontWeight: "800",
    letterSpacing: -1,
  },
});
