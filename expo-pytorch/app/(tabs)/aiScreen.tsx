import { Text, View, StyleSheet, ScrollView, Image, Pressable } from "react-native";
import { useState } from "react";
import { useClassification, EFFICIENTNET_V2_S } from 'react-native-executorch';
import { AlbumContext } from "../../AlbumContext";
import { useContext, useEffect} from "react";
import * as MediaLibrary from 'expo-media-library';
import { SaveFormat, useImageManipulator } from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system';


/*export async function getLocalUri(asset: MediaLibrary.Asset) {
// Because of how ios and expo-media-library work, we can't use a URI, we need an actual file path. 
// Just store the image in app cache and then return the image path. precious hack

  // For assets that are already file://, just return
  if (asset.uri.startsWith('file://') && !asset.uri.toLowerCase().endsWith('.heic')) return asset.uri;

  // Get asset info
  const info = await MediaLibrary.getAssetInfoAsync(asset.id);
  if (!info.localUri) throw new Error(`Cannot get local URI for asset ${asset.id}`);

  let localUri = info.localUri

  if (localUri.toLowerCase().endsWith('.heic')) {
    const manipulator = useImageManipulator(localUri);
    const result = (await manipulator.renderAsync()).saveAsync({
        format: SaveFormat.JPEG,
        base64: false, 
        compress: 1,
    });
    localUri = (await result).uri;

  }

  return info.localUri; // this is file://
}*/


type ImageResult = {
  uri: string;
  top10: { label: string; score: number }[];
};

export default function AiScreen(){
    const { assets } = useContext(AlbumContext);
    const [imageResults, setImageResults] = useState<ImageResult[]>([]);
    
    const [isRunning, setIsRunning] = useState(false);

    const model = useClassification({ model: EFFICIENTNET_V2_S }); // get the model

    useEffect(() => {
        if (!model.isReady || assets.length === 0 || !isRunning) return;
    
        const classifyAlbum = async () => {
            const newResults: ImageResult[] = [];
        
            for (const asset of assets) {
                try {
                    const localUri = asset.uri

                    console.log("Debug: ", localUri)
                    
                    const output = await model.forward(localUri);

                    const top10 = Object.entries(output)
                        .sort(([, a], [, b]) => (b as number) - (a as number))
                        .slice(0, 10)
                        .map(([label, score]) => ({ label, score: score as number }));

                    newResults.push({ uri: asset.uri, top10 });

                    // update state incrementally so UI shows progress
                    setImageResults([...newResults]);
                } catch (e) {
                    console.error(`Error classifying ${asset.uri}:`, e);
                }
            }
        };
        classifyAlbum();
    }, [model.isReady, assets, isRunning]);

  if (!model.isReady) {
    return (
      <View style={styles.center}>
        <Text>Loading model...</Text>
      </View>
    );
  }

  if (!isRunning){
    return(
        <Pressable style={styles.buttonContainer} onPress={() => setIsRunning(true)}>
              <Text style={styles.button}>Start profiling (maybe?)</Text>
        </Pressable>
    );
  }
  else{
    return (
    <ScrollView contentContainerStyle={styles.container}>
      {imageResults.map(({ uri, top10 }) => (
        <View key={uri} style={styles.imageBlock}>
          <Image source={{ uri }} style={styles.image} resizeMode="contain" />
          <Text style={styles.header}>Top 10 Results:</Text>
          {top10.map(({ label, score }) => (
            <Text key={label}>{label}: {score.toFixed(3)}</Text>
          ))}
        </View>
      ))}
      </ScrollView>
    );
  }
}


const styles = StyleSheet.create({
  container: {
    padding: 16,
  },
  imageBlock: {
    marginBottom: 24,
    borderBottomWidth: 1,
    borderColor: "#ccc",
    paddingBottom: 12,
  },
  image: {
    width: "100%",
    height: 200,
    borderRadius: 8,
    marginBottom: 8,
  },
  header: {
    fontWeight: "bold",
    marginBottom: 4,
  },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
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
});