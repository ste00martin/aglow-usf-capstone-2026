import { useEffect, useState } from "react";
import { Alert, Pressable, ScrollView, Text, View, Linking, StyleSheet, Button, TouchableOpacity, TurboModuleRegistry, ActivityIndicator } from "react-native";
import { Image } from 'expo-image';
import { AlbumContext } from "../../AlbumContext";
import { useContext } from "react";
import * as ImagePicker from 'expo-image-picker';
import { useVideoPlayer, VideoView } from 'expo-video';
import * as VideoThumbnails from 'expo-video-thumbnails';
import { useExecutorchModule, ScalarType, useSpeechToText, WHISPER_TINY } from "react-native-executorch";
import { extractAudio } from 'expo-video-audio-extractor';
import { AudioContext } from 'react-native-audio-api';
import { useAudioPlayer, useAudioPlayerStatus } from 'expo-audio';
import * as FileSystem from 'expo-file-system/legacy';
import type { TensorPtr } from "react-native-executorch";
import { imageUriToViTTensor, allFromLogits, allFromSigmoid } from "../../aipreprocessing"
import { env, AutoTokenizer } from '@xenova/transformers';

env.allowLocalModels = false;
env.useBrowserCache = false;

import { NSFW_MODEL, TEXT_MODEL } from "../../assets/models/executorchModels";
const VIDEO_MAX_DURATION = 30; // seconds
const FRAME_INTERVAL = 1000; // milliseconds between frames to extract
const NSFW_LABELS = ['gore_bloodshed_violent', 'nudity_pornography', 'safe_normal'];
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
    const [audio, setAudio] = useState<string | null>(null);
    const [video, setVideo] = useState<string>("");
    const [thumbnails, setThumbnails] = useState<ImageResult[]>([]);
    const [flagged, setFlagged] = useState<ImageResult[]>([])
    const [transcript, setTranscript] = useState<string | null>("");
    const [textModeration, setTextModeration] = useState<{ label: string, score: number }[]>([]);

    const [running, setRunning] = useState(false);
    const [videoLoaded, setVideoLoaded] = useState(false);

    const [totalExpanded, setTotalExpanded] = useState(false);
    const [flaggedExpanded, setFlaggedExpanded] = useState(false);
    const [transcriptExpanded, setTranscriptExpanded] = useState(false);
    const [audioButtonStatus, setAudioButtonStatus] = useState<boolean>(false);
    const [replayButtonStatus, setReplayButtonStatus] = useState<boolean>(false);

    const nsfwModel = useExecutorchModule({ modelSource: NSFW_MODEL });
    const ttsModel = useSpeechToText({ model: WHISPER_TINY, });
    const textModel = useExecutorchModule({ modelSource: TEXT_MODEL });

    const isReady = nsfwModel.isReady && ttsModel.isReady && textModel.isReady;

    useEffect(() => {
        if (!isReady || !running) return;
        const pickVideo = async () => {
            setThumbnails([]); // reset thumbnails when picking a new video
            setFlagged([]); // reset flagged results when picking a new video
            setTextModeration([]); // reset text moderation results
            setVideo(""); // reset video URI
            setAudio(null); // reset audio URI

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
                const selectedUri = result.assets[0].uri;
                const duration = result.assets[0].duration ?? 30000;

                console.log(FileSystem);

                const outputUri = FileSystem.documentDirectory + 'speech.m4a';

                try {
                    await FileSystem.deleteAsync(outputUri, { idempotent: true });
                } catch (e) { }

                try {
                    await extractAudio({
                        // Required
                        video: selectedUri,
                        output: outputUri,

                        // Optional controls ↓
                        format: 'm4a',      // 'm4a' (default) or 'wav'
                        volume: 0.9,        // 90 % volume (linear gain)
                        channels: 2,        // force mono (wav only)
                        sampleRate: 16000,  // override sample-rate (wav only)
                    });
                } catch (e) {
                    console.error("extractAudio error:", e);
                }

                console.log('Audio saved at', outputUri);
                setAudio(outputUri);
                const transcribedAudio = await transcribeAudio(outputUri);
                setTranscript(transcribedAudio)

                if (transcribedAudio) {
                    console.log("Transcribed Audio:", transcribedAudio);
                    try {
                        // unitary/toxic-bert uses standard bert-base-uncased tokenizer
                        // which Xenova supports natively — no special token remapping needed.
                        const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased');
                        const tokens = await tokenizer(transcribedAudio, {
                            padding: 'max_length',
                            truncation: true,
                            maxLength: 128,
                        });

                        const inputIdsData = tokens.input_ids.data instanceof BigInt64Array
                            ? tokens.input_ids.data
                            : new BigInt64Array(tokens.input_ids.data);

                        const attentionMaskData = tokens.attention_mask.data instanceof BigInt64Array
                            ? tokens.attention_mask.data
                            : new BigInt64Array(tokens.attention_mask.data);

                        const inputIdsPtr: TensorPtr = {
                            dataPtr: inputIdsData,
                            sizes: [1, 128],
                            scalarType: ScalarType.LONG,
                        };

                        const attentionMaskPtr: TensorPtr = {
                            dataPtr: attentionMaskData,
                            sizes: [1, 128],
                            scalarType: ScalarType.LONG,
                        };

                        const textOutputs = await textModel.forward([inputIdsPtr, attentionMaskPtr]);
                        const rawData = textOutputs[0].dataPtr;
                        const textLogits = rawData instanceof Float32Array
                            ? rawData
                            : rawData instanceof ArrayBuffer
                                ? new Float32Array(rawData)
                                : Float32Array.from(rawData as unknown as number[]);
                        console.log("textLogits: ", textLogits.length, Array.from(textLogits));
                        // unitary/toxic-bert: 6-label multi-label sigmoid (Jigsaw Toxic Comment categories)
                        const textLabels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'];
                        const textResult = allFromSigmoid(textLogits, textLabels);

                        setTextModeration(textResult);
                        console.log("Text Moderation complete:", textResult);
                    } catch (e) {
                        console.error("Text moderation inference failed: ", e);
                    }
                }

                console.log("Duration: " + duration);

                const thumbnails: ImageResult[] = [];
                const flagged: ImageResult[] = [];

                setVideo(result.assets[0].uri); // always chooses the first video
                for (let time = 0; time < duration; time += FRAME_INTERVAL) {
                    let thumbnail: string;
                    try {
                        const res = await VideoThumbnails.getThumbnailAsync(selectedUri, { time: time });
                        thumbnail = res.uri;
                    } catch (e) {
                        console.log(`Failed to extract frame at ${time}ms (hit EOF boundary), breaking loop.`);
                        break;
                    }

                    console.log(`Extracted frame at ${time}ms: ${thumbnail}`);

                    let nonCroppedVitTensor;
                    try {
                        nonCroppedVitTensor = await imageUriToViTTensor(thumbnail);
                    } catch (e) {
                        console.error(`Image manipulation failed at frame ${time}ms:`, e);
                        continue;
                    }
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
        pickVideo()
    }, [running]);


    const transcribeAudio = async (audioUri: string) => {
        const response = await fetch(audioUri);
        const arrayBuffer = await response.arrayBuffer();

        const audioContext = new AudioContext({ sampleRate: 16000 });
        const decodedAudioData = await audioContext.decodeAudioData(arrayBuffer);
        const audioBuffer = decodedAudioData.getChannelData(0);

        try {
            const transcription = await ttsModel.transcribe(audioBuffer, { language: 'en' });
            return transcription;
        } catch (error) {
            console.error('Error during audio transcription', error);
            return null;
        }
    }
    const audioPlayer = useAudioPlayer(
        { uri: audio ?? undefined }
    );
    const player = useVideoPlayer(
        { uri: video ?? undefined } // this accepts your remote URI directly
    );

    const status = useAudioPlayerStatus(audioPlayer);

    if (!isReady) {
        return (
            <View style={[styles.container, { flex: 1, justifyContent: 'center' }]}>
                <ActivityIndicator size="large" color="#0000ff" />
                <Text style={{ marginTop: 10 }}>Loading models...</Text>
            </View>
        );
    }

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
                        <Pressable style={styles.buttonContainer} onPress={() => setRunning(true)}>
                            <Text style={styles.button}>Upload a Video.</Text>
                        </Pressable>
                    </>
                ) : (
                    <>
                        {!running && (
                            <>
                                <Pressable style={styles.buttonContainer} onPress={() => setRunning(true)}>
                                    <Text style={styles.button}>Upload another Video.</Text>
                                </Pressable>
                                <VideoView
                                    player={player}
                                    style={styles.video}
                                    nativeControls
                                    contentFit="contain"
                                />
                            </>
                        )}

                        {!running && audio != null && (
                            <View key={audio} style={{ marginVertical: 10 }}>
                                {audioButtonStatus === false ? (
                                    <>
                                        <Pressable
                                            style={styles.buttonContainer}
                                            onPress={() => {
                                                if (!audioPlayer || !status) return;

                                                // Compare current time with duration
                                                if (status.currentTime < status.duration) {
                                                    audioPlayer.play(); // resume if not finished
                                                } else {
                                                    audioPlayer.seekTo(0); // reset and play if finished
                                                    audioPlayer.play();
                                                }
                                                setAudioButtonStatus(true);
                                                setFlaggedExpanded(false);
                                                setTotalExpanded(false);
                                            }}
                                        >
                                            <Text style={styles.button}>Play Audio</Text>
                                        </Pressable>
                                    </>
                                ) : (
                                    <>
                                        <Pressable
                                            style={styles.buttonContainer}
                                            onPress={() => {
                                                audioPlayer.pause()
                                                setAudioButtonStatus(false)
                                            }}
                                        >
                                            <Text style={styles.button}>Pause Audio</Text>
                                        </Pressable>
                                    </>
                                )}
                                <Text>Playing: {status.playing ? 'Yes' : 'No'}</Text>
                                <Text>Current Time: {status.currentTime}s</Text>
                                <Text>Duration: {status.duration}s</Text>
                            </View>

                        )}

                        {!running && audio != null && (
                            <View style={{ marginVertical: 10 }}>
                                <Pressable
                                    style={styles.buttonContainer}
                                    onPress={() => {
                                        if (!audioPlayer) return;

                                        audioPlayer.seekTo(0); // reset to start
                                        audioPlayer.play();    // immediately play
                                        setAudioButtonStatus(true); // show the pause button
                                    }}
                                >
                                    <Text style={styles.button}>Replay Audio</Text>
                                </Pressable>
                            </View>
                        )}

                        {flagged.length > 0 && (
                            <>
                                {!flaggedExpanded ? (
                                    <>
                                        {flagged.length > 0 && (
                                            <Pressable style={styles.buttonContainer} onPress={() => {
                                                setFlaggedExpanded(!flaggedExpanded)
                                                setTotalExpanded(false);
                                            }}>
                                                <Text style={styles.button}>Open Flagged Analysis</Text>
                                            </Pressable>
                                        )}
                                    </>
                                ) : (
                                    <>
                                        <Pressable style={styles.buttonContainer} onPress={() => {
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
                                    <Pressable style={styles.buttonContainer} onPress={() => {
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

                        {(!transcriptExpanded) ? (
                            <>
                                {thumbnails.length > 0 && (
                                    <Pressable style={styles.buttonContainer} onPress={() => {
                                        setTranscriptExpanded(!transcriptExpanded)
                                        setFlaggedExpanded(false);
                                    }}>
                                        <Text style={styles.button}>Open Transcript</Text>
                                    </Pressable>
                                )}
                            </>
                        ) : (
                            <>
                                <Pressable style={styles.buttonContainer} onPress={() => setTranscriptExpanded(!transcriptExpanded)}>
                                    <Text style={styles.button}>Close Transcript</Text>
                                </Pressable>
                                <Text style={styles.text}>
                                    {transcript || "No transcription yet..."}
                                </Text>
                                {textModeration.length > 0 && (
                                    <View style={{ marginTop: 20, alignItems: 'center', width: '100%' }}>
                                        <Text style={{ fontWeight: 'bold', fontSize: 18, marginBottom: 10 }}>Text Moderation Results:</Text>
                                        <View style={{ backgroundColor: '#f0f0f0', padding: 15, borderRadius: 10, width: '90%' }}>
                                            {textModeration.map((res, i) => {
                                                const isFlagged = res.score > 0.4;
                                                return (
                                                    <View key={i} style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 5 }}>
                                                        <Text style={{ color: isFlagged ? 'red' : 'black', fontWeight: isFlagged ? 'bold' : 'normal' }}>
                                                            {res.label}
                                                        </Text>
                                                        <Text style={{ color: isFlagged ? 'red' : 'black' }}>
                                                            {(res.score * 100).toFixed(1)}%
                                                        </Text>
                                                    </View>
                                                );
                                            })}
                                        </View>
                                    </View>
                                )}
                            </>
                        )}
                    </>
                )}
            </ScrollView>
        </View>
    );
}


const styles = StyleSheet.create({
    button: {
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
        aspectRatio: 16 / 9,
        marginVertical: 20,
    },
    text: {
        fontSize: 16,
    },
});
