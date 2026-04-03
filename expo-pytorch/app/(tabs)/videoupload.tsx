import { useEffect, useState } from "react";
import { Alert, Pressable, Image, ScrollView, Text, View, Linking, StyleSheet, Button, TouchableOpacity, TurboModuleRegistry, ActivityIndicator } from "react-native";
import { AlbumContext } from "../../AlbumContext";
import { useContext } from "react";
import * as ImagePicker from 'expo-image-picker';
import { useVideoPlayer, VideoView } from 'expo-video';
import * as VideoThumbnails from 'expo-video-thumbnails';
import { useExecutorchModule, ScalarType } from "react-native-executorch";
import type { TensorPtr } from "react-native-executorch";
import { imageUriToTensor, postprocessBlazeFace, cropFace, imageUriToViTTensor, topFromLogits, allFromLogits } from "../../aipreprocessing"
import { NSFW_MODEL } from "../../assets/models/executorchModels";

const VIDEO_MAX_DURATION = 30; // seconds
const FRAME_INTERVAL = 1000; // milliseconds between frames to extract
const NSFW_LABELS   = ['gore_bloodshed_violent', 'nudity_pornography', 'safe_normal'];
const NSFW_METRIC = 0.5; // threshold for flagging content as NSFW

const VIT_INPUT_SIZE = 224;

type ImageResult = {
    timestamp: number;
    uri: string;
    nsfw: { label: string; score: number }[];
};

function formatTime(ms: number): string { // applies flooring by default, e.g. 5500ms is 0:05, not 0:06
    const totalSeconds = Math.floor(ms / 1000); // convert milliseconds to seconds
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const paddedSeconds = seconds.toString().padStart(2, '0'); // ensures 2 digits
    return `${minutes}:${paddedSeconds}`;
}

export default function VideoUploadScreen() {
    // some code here.
    const [video, setVideo] = useState<string>("");
    const [thumbnails, setThumbnails] = useState<ImageResult[]>([]);
    const [flagged, setFlagged] = useState<ImageResult[]>([])

    const [running, setRunning] = useState(false);
    const [videoLoaded, setVideoLoaded] = useState(false); 

    const [totalExpanded, setTotalExpanded] = useState(false);
    const [flaggedExpanded, setFlaggedExpanded] = useState(false);

    const nsfwModel = useExecutorchModule({ modelSource: NSFW_MODEL });


    const pickVideo = async () => {
        setThumbnails([]); // reset thumbnails when picking a new video
        setFlagged([]); // reset flagged results when picking a new video
        setVideo(""); // reset video URI
        setRunning(false);

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
            videoMaxDuration: VIDEO_MAX_DURATION, // limit to VIDEO_MAX_DURATION seconds
        });
        
        if (!result.canceled) { // if a video is loaded successfully..
            setVideoLoaded(true)
            setRunning(true);
            const selectedUri = result.assets[0].uri;
            const duration = result.assets[0].duration ?? 30000;

            console.log("Duration: " + duration);

            const thumbnails: ImageResult[] = [];
            const flagged: ImageResult[] = [];

            setVideo(result.assets[0].uri); // always chooses the first video
            for (let time = 0; time < duration; time += FRAME_INTERVAL) {
                const { uri: thumbnail } = await VideoThumbnails.getThumbnailAsync(selectedUri, { time: time });
                
                console.log(`Extracted frame at ${time}ms: ${thumbnail}`);

                const nonCroppedVitTensor = await imageUriToViTTensor(thumbnail);
                const nonCroppedVitTensorPtr: TensorPtr = {
                    dataPtr: nonCroppedVitTensor,
                    sizes: [1, 3, VIT_INPUT_SIZE, VIT_INPUT_SIZE],
                    scalarType: ScalarType.FLOAT,
                };

                const nsfwOutputs = await nsfwModel.forward([nonCroppedVitTensorPtr]);

                const nsfwLogits = new Float32Array(nsfwOutputs[0].dataPtr as ArrayBuffer);
                const result = allFromLogits(nsfwLogits, NSFW_LABELS);

                thumbnails.push({
                    timestamp: time,
                    uri: thumbnail,
                    nsfw: result,
                })

                const flag = result.some(
                    (n) =>
                        (n.label === "gore_bloodshed_violent" || n.label === "nudity_pornography") &&
                        n.score >= NSFW_METRIC
                );
                if (flag) {
                    flagged.push({
                        timestamp: time,
                        uri: thumbnail,
                        nsfw: result,
                    })
                }
            }
            setThumbnails([...thumbnails])
            setFlagged([...flagged])
            setRunning(false);
        }
    };
    const player = useVideoPlayer(
        { uri: video ?? undefined } // this accepts your remote URI directly
    );

    return (
        <View style={{ flex: 1 }}>
            {running && (
                <View style={styles.overlay}>
                    <ActivityIndicator size="large" color="#fff" />
                </View>
            )}
          <ScrollView
            contentContainerStyle={styles.container}
            showsVerticalScrollIndicator={true}
          >
            {!videoLoaded ? (
                <>
                    <Pressable style={styles.buttonContainer} onPress={pickVideo}>
                        <Text style={styles.button}>Upload a Video.</Text>
                    </Pressable>
                </>
            ) : (
                <>
                    <Pressable style={styles.buttonContainer} onPress={pickVideo}>
                        <Text style={styles.button}>Upload another Video.</Text>
                    </Pressable>
                    <VideoView
                        player={player}
                        style={styles.video}
                        nativeControls
                        contentFit="contain"
                    />
                    {flagged.length > 0 && (
                        <>
                            {!flaggedExpanded ? (
                                <>
                                    {flagged.length > 0 && (
                                        <Pressable style={styles.buttonContainer} onPress={() => 
                                        {   
                                            setFlaggedExpanded(!flaggedExpanded)
                                            setTotalExpanded(false);
                                        }}>
                                            <Text style={styles.button}>Open Flagged Analysis</Text>
                                        </Pressable>
                                    )}
                                </>
                            ): (
                                <>
                                <Pressable style={styles.buttonContainer} onPress={() => 
                                    {
                                        setFlaggedExpanded(!flaggedExpanded)
                                    }}>
                                    <Text style={styles.button}>Close Flagged Analysis</Text>
                                </Pressable>
                                {flagged.map((item, index) => (
                                    <View key={index} style={{ marginVertical: 15, alignItems: 'center' }}>
                                        <Text>Timestamp: {formatTime(item.timestamp)} </Text>
                                        <Image
                                            source={{ uri: item.uri }}
                                            style={{ width: 200, height: 120 }}
                                        />
                                        <View>
                                            {item.nsfw.map((n, i) => (
                                                <Text key={i}>
                                                    {n.label} ({(n.score * 100).toFixed(1)}%)
                                                </Text>
                                        ))}
                                        </View>
                                    </View>
                                ))}
                            </>
                        )}
                        </>
                    )}

                    {(!totalExpanded) ? (
                        <>
                            {thumbnails.length > 0 && (
                                <Pressable style={styles.buttonContainer} onPress={() => 
                                {
                                    setTotalExpanded(!totalExpanded)
                                    setFlaggedExpanded(false);
                                }}>
                                    <Text style={styles.button}>Open Frame Analysis</Text>
                                </Pressable>
                            )}
                        </>
                    ) : (
                        <>
                            <Pressable style={styles.buttonContainer} onPress={() => setTotalExpanded(!totalExpanded)}>
                                <Text style={styles.button}>Close Frame Analysis</Text>
                            </Pressable>
                            {thumbnails.map((item, index) => (
                                <View key={index} style={{ marginVertical: 15, alignItems: 'center' }}>
                                    <Text>Timestamp: {formatTime(item.timestamp)}</Text>
                                    <Image
                                        source={{ uri: item.uri }}
                                        style={{ width: 200, height: 120 }}
                                    />
                                    <View>
                                        {item.nsfw.map((n, i) => (
                                            <Text key={i}>
                                                {n.label} ({(n.score * 100).toFixed(1)}%)
                                            </Text>
                                        ))}
                                    </View>
                                </View>
                            ))}
                        </>
                    )}
                </>
            )}
          </ScrollView>
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
        alignItems: 'center',
        paddingVertical: 10,
    },
    image: {
        width: 200,
        height: 200,
    },
    overlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(0,0,0,0.5)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    video: { 
        width: '100%', 
        aspectRatio: 16/9,
        marginVertical: 20,
    },
});
