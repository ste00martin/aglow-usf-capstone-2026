import { Stack } from "expo-router";
import { WorkflowProvider } from "../AlbumContext";
import { workflowTheme } from "@/components/workflow/theme";

export default function RootLayout() {
  return (
    <WorkflowProvider>
      <Stack screenOptions={{ contentStyle: { backgroundColor: workflowTheme.background }, headerShown: false }}>
        <Stack.Screen name="index" />
        <Stack.Screen name="review" />
        <Stack.Screen name="processing" />
        <Stack.Screen name="results" />
        <Stack.Screen name="result/[assetId]" />
      </Stack>
    </WorkflowProvider>
  );
}
