import { useEffect, useState } from "react";
import { Alert, Pressable, Image, ScrollView, Text, View, Linking, StyleSheet, Button} from "react-native";
import { AlbumContext } from "../../AlbumContext";
import { useContext } from "react";
import * as ImagePicker from 'expo-image-picker';
import { useVideoPlayer, VideoView } from 'expo-video';

export default function VideoUploadScreen() {
    // some code here.
    const [video, setVideo] = useState<string | null>(null);

    const pickVideo = async () => {
        const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (!permissionResult.granted) {
            Alert.alert('Permission required', 'Permission to access the media library is required.');
            return;
        }

        let result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ['videos'],
            allowsEditing: true,
            aspect: [4, 3],
            quality: 1,
        });
        
        if (!result.canceled) { // if we get something..
            setVideo(result.assets[0].uri); // always chooses the first video
        }
    };

    const player = useVideoPlayer(
        { uri: video ?? undefined } // this accepts your remote URI directly
    );

    return (
        <View style={styles.container}>
            <Pressable style={styles.buttonContainer} onPress={pickVideo}>
                <Text style={styles.button}>Upload a Video.</Text>
            </Pressable>                
            <VideoView
                player={player}
                style={styles.video}
                nativeControls        // show native play/pause controls
                contentFit="contain" // keeps aspect ratio
            />
        </View>
    );
}


const styles = StyleSheet.create({
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

    container: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    image: {
        width: 200,
        height: 200,
    },
    video: { width: '100%', aspectRatio: 16/9 },
});