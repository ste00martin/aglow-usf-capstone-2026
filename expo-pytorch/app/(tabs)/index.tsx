import { useCallback, useContext } from "react";
import { Alert, Pressable, ScrollView, Text, View, Linking, StyleSheet, useWindowDimensions } from "react-native";
import { Image } from 'expo-image';
import { AlbumContext, type AlbumAsset } from "../../AlbumContext";
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';


async function albumAssetFromPickerAsset(
  asset: ImagePicker.ImagePickerAsset,
  index: number
): Promise<AlbumAsset> {
  let uri = asset.uri;

  let width = asset.width > 0 ? asset.width : undefined;
  let height = asset.height > 0 ? asset.height : undefined;

  if (uri.toLowerCase().endsWith(".heic")) {
    const result = await ImageManipulator.manipulateAsync(
      uri,
      [],
      { format: ImageManipulator.SaveFormat.JPEG, compress: 1 }
    );
    uri = result.uri;
    width = result.width;
    height = result.height;
  }

  const id = asset.assetId ?? `pick-${index}-${uri}`;
  return {
    id,
    uri,
    mediaType: "photo",
    ...(width !== undefined ? { width } : {}),
    ...(height !== undefined ? { height } : {}),
  };
}


export default function HomeScreen() {
  const [permissionResponse, requestPermission] = ImagePicker.useMediaLibraryPermissions();
  const { assets, setAssets } = useContext(AlbumContext);
  const { width: windowWidth } = useWindowDimensions();
  const horizontalPadding = 16;
  const columnGap = 8;
  const columnWidth = (windowWidth - horizontalPadding * 2 - columnGap) / 2;

  const mergePickerResults = useCallback(
    async (picked: ImagePicker.ImagePickerAsset[]) => {
      const newItems = await Promise.all(
        picked.map((a, i) => albumAssetFromPickerAsset(a, i))
      );
      setAssets((prev) => {
        const seen = new Set(prev.map((p) => p.uri));
        const next = [...prev];
        for (const item of newItems) {
          if (!seen.has(item.uri)) {
            seen.add(item.uri);
            next.push(item);
          }
        }
        return next;
      });
    },
    [setAssets]
  );

  const pickFromLibrary = async () => {
    const perm =
      permissionResponse?.status === "granted"
        ? permissionResponse
        : await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      Alert.alert(
        "Permission required",
        "Photo library access is needed to choose images."
      );
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      allowsMultipleSelection: true,
      selectionLimit: 0,
      quality: 1,
    });

    if (result.canceled || !result.assets?.length) return;
    await mergePickerResults(result.assets);
  };

  return (
    <ScrollView contentContainerStyle={styles.scrollContent}>
      <Text style={styles.heading}>
        Image Selection
      </Text>

      {permissionResponse?.status !== "granted" ? (
        <Pressable style={styles.buttonContainer} onPress={requestPermission}>
          <Text style={styles.button}>Get Started (maybe?)</Text>
        </Pressable>
      ) : (
        <>
          <View style={styles.toolbar}>
            <Pressable style={styles.buttonContainer} onPress={pickFromLibrary}>
              <Text style={styles.button}>Add from library</Text>
            </Pressable>

            <Pressable style={styles.buttonContainer} onPress={() => Linking.openSettings()}>
              <Text style={styles.button}>Change Access</Text>
            </Pressable>
          </View>

          <View style={[styles.grid, { gap: columnGap }]}>
            {assets.map((asset) => {
              if (asset.mediaType !== "photo") {
                return null;
              }
              const cellStyle = { width: columnWidth };
              const imageStyle = [
                styles.mediaItem,
                { width: columnWidth, height: columnWidth },
              ];
              return (
                <View key={asset.id} style={cellStyle}>
                  <Image source={{ uri: asset.uri }} style={imageStyle} />
                </View>
              );
            })}
          </View>
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
    scrollContent: {
      padding: 16,
      alignItems: 'stretch',
    },
    heading: {
      fontSize: 18,
      fontWeight: 'bold',
      textAlign: 'center',
      marginBottom: 8,
    },
    toolbar: {
      flexDirection: 'column',
      alignItems: 'center',
      marginBottom: 4,
    },
    grid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      width: '100%',
    },
    button:{
      fontSize: 20,
      color: '#fff',
    },
    buttonContainer: {
      paddingVertical: 10,
      paddingHorizontal: 16,
      borderRadius: 8,
      backgroundColor: '#333',
      margin: 8,
      alignItems: 'center',
    },
    mediaItem: {
      borderRadius: 8,
    },
});
