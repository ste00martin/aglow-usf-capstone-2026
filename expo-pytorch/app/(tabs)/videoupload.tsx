import { useEffect, useMemo, useState } from "react";
import { Alert, Pressable, ScrollView, Text, View, StyleSheet, ActivityIndicator } from "react-native";
import { Image } from 'expo-image';
import { Ionicons } from "@expo/vector-icons";
import { SafeAreaView } from "react-native-safe-area-context";
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

type ContentSection = "video" | "audio" | "frames" | "flagged" | "transcript";

const SECTION_LABELS: Record<ContentSection, string> = {
    video: "Video",
    audio: "Extracted audio",
    frames: "Frame analysis",
    flagged: "Flagged frames",
    transcript: "Transcript",
};

const DRAWER_ICONS: Record<ContentSection, keyof typeof Ionicons.glyphMap> = {
    video: "videocam-outline",
    audio: "musical-notes-outline",
    frames: "images-outline",
    flagged: "flag-outline",
    transcript: "document-text-outline",
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
    const [drawerOpen, setDrawerOpen] = useState(false);
    const [activeSection, setActiveSection] = useState<ContentSection>("video");
    const [audioButtonStatus, setAudioButtonStatus] = useState<boolean>(false);

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
            setAudioButtonStatus(false);
            setActiveSection("video");
            setDrawerOpen(false);

            const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
            if (!permissionResult.granted) {
                Alert.alert('Permission required', 'Permission to access the media library is required.');
                setRunning(false);
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
            } else {
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

    const menuSections = useMemo((): ContentSection[] => {
        const sections: ContentSection[] = ["video", "audio", "frames"];
        if (flagged.length > 0) sections.push("flagged");
        sections.push("transcript");
        return sections;
    }, [flagged.length]);

    useEffect(() => {
        if (activeSection === "flagged" && flagged.length === 0) {
            setActiveSection("video");
        }
    }, [flagged.length, activeSection]);

    const headerTitle = !videoLoaded
        ? "Video Upload"
        : running
            ? "Processing…"
            : SECTION_LABELS[activeSection];

    if (!isReady) {
        return (
            <View style={[styles.container, { flex: 1, justifyContent: 'center' }]}>
                <ActivityIndicator size="large" color="#0000ff" />
                <Text style={{ marginTop: 10 }}>Loading models...</Text>
            </View>
        );
    }

    const renderSectionBody = () => {
        if (!videoLoaded || running) return null;

        switch (activeSection) {
            case "video":
                return (
                    <View style={styles.sectionBlock}>
                        <VideoView
                            player={player}
                            style={styles.video}
                            nativeControls
                            contentFit="contain"
                        />
                    </View>
                );
            case "audio":
                if (audio == null) {
                    return (
                        <Text style={styles.mutedCenter}>No extracted audio for this clip.</Text>
                    );
                }
                return (
                    <View key={audio} style={styles.sectionBlock}>
                        <View style={styles.audioControlsRow}>
                            {audioButtonStatus ? (
                                <Pressable
                                    style={styles.buttonContainer}
                                    onPress={() => {
                                        audioPlayer.pause();
                                        setAudioButtonStatus(false);
                                    }}
                                >
                                    <Text style={styles.button}>Pause</Text>
                                </Pressable>
                            ) : (
                                <Pressable
                                    style={styles.buttonContainer}
                                    onPress={() => {
                                        if (!audioPlayer || !status) return;
                                        if (status.currentTime < status.duration) {
                                            audioPlayer.play();
                                        } else {
                                            audioPlayer.seekTo(0);
                                            audioPlayer.play();
                                        }
                                        setAudioButtonStatus(true);
                                    }}
                                >
                                    <Text style={styles.button}>Play</Text>
                                </Pressable>
                            )}
                            <Pressable
                                style={styles.buttonContainer}
                                onPress={() => {
                                    if (!audioPlayer) return;
                                    audioPlayer.seekTo(0);
                                    audioPlayer.play();
                                    setAudioButtonStatus(true);
                                }}
                            >
                                <Text style={styles.button}>Replay</Text>
                            </Pressable>
                        </View>
                        <View style={styles.audioMeta}>
                            <Text style={styles.metaLine}>Playing: {status.playing ? "Yes" : "No"}</Text>
                            <Text style={styles.metaLine}>
                                Time: {status.currentTime.toFixed(1)}s / {status.duration.toFixed(1)}s
                            </Text>
                        </View>
                    </View>
                );
            case "frames":
                if (thumbnails.length === 0) {
                    return (
                        <Text style={styles.mutedCenter}>No frame analysis yet.</Text>
                    );
                }
                return (
                    <View style={styles.sectionBlock}>
                        {thumbnails.map((item) => (
                            <FrameResultCard item={item} key={`${item.timestamp}-${item.uri}`} />
                        ))}
                    </View>
                );
            case "flagged":
                if (flagged.length === 0) {
                    return (
                        <Text style={styles.mutedCenter}>No flagged frames for this video.</Text>
                    );
                }
                return (
                    <View style={styles.sectionBlock}>
                        {flagged.map((item) => (
                            <FrameResultCard item={item} key={`${item.timestamp}-${item.uri}`} />
                        ))}
                    </View>
                );
            case "transcript":
                return (
                    <View style={styles.sectionBlock}>
                        <Text style={styles.transcriptText}>
                            {transcript || "No transcription yet…"}
                        </Text>
                        {textModeration.length > 0 && (
                            <View style={styles.moderationBlock}>
                                <Text style={styles.moderationHeading}>Text moderation</Text>
                                <View style={styles.moderationCard}>
                                    {textModeration.map((res, i) => {
                                        const isFlaggedText = res.score > 0.4;
                                        return (
                                            <View key={res.label + i} style={styles.moderationRow}>
                                                <Text
                                                    style={[
                                                        styles.moderationLabel,
                                                        isFlaggedText && styles.moderationLabelFlagged,
                                                    ]}
                                                >
                                                    {res.label}
                                                </Text>
                                                <Text
                                                    style={[
                                                        styles.moderationScore,
                                                        isFlaggedText && styles.moderationScoreFlagged,
                                                    ]}
                                                >
                                                    {(res.score * 100).toFixed(1)}%
                                                </Text>
                                            </View>
                                        );
                                    })}
                                </View>
                            </View>
                        )}
                    </View>
                );
            default:
                return null;
        }
    };

    return (
        <View style={styles.screenRoot}>
            {running && (
                <View style={styles.overlay}>
                    <ActivityIndicator size="large" color="#fff" />
                </View>
            )}
            <SafeAreaView edges={["top"]} style={styles.headerSafe}>
                <View style={styles.headerRow}>
                    <View style={styles.headerSideSlot}>
                        {videoLoaded && !running ? (
                            <Pressable
                                onPress={() => setDrawerOpen(true)}
                                hitSlop={10}
                                accessibilityRole="button"
                                accessibilityLabel="Open contents menu"
                            >
                                <Ionicons name="menu" size={26} color="#fff" />
                            </Pressable>
                        ) : null}
                    </View>
                    <Text style={styles.headerTitleText} numberOfLines={1}>
                        {headerTitle}
                    </Text>
                    <View style={styles.headerSideSlot}>
                        {videoLoaded && !running ? (
                            <Pressable
                                onPress={() => setRunning(true)}
                                hitSlop={10}
                                accessibilityRole="button"
                                accessibilityLabel="Upload another video"
                            >
                                <Ionicons name="add-circle-outline" size={26} color="#fff" />
                            </Pressable>
                        ) : null}
                    </View>
                </View>
            </SafeAreaView>

            <ScrollView
                contentContainerStyle={styles.bodyScroll}
                showsVerticalScrollIndicator
            >
                {!videoLoaded ? (
                    <Pressable style={styles.buttonContainer} onPress={() => setRunning(true)}>
                        <Text style={styles.button}>Upload a Video</Text>
                    </Pressable>
                ) : (
                    renderSectionBody()
                )}
            </ScrollView>

            {drawerOpen && videoLoaded && !running ? (
                <View style={styles.drawerRoot} pointerEvents="box-none">
                    <Pressable
                        style={styles.drawerBackdrop}
                        onPress={() => setDrawerOpen(false)}
                        accessibilityRole="button"
                        accessibilityLabel="Close menu"
                    />
                    <SafeAreaView edges={["top", "left", "bottom"]} style={styles.drawerSheet}>
                        <Text style={styles.drawerHeading}>Contents</Text>
                        {menuSections.map((section) => {
                            const active = activeSection === section;
                            return (
                                <Pressable
                                    key={section}
                                    style={[styles.drawerItem, active && styles.drawerItemActive]}
                                    onPress={() => {
                                        setActiveSection(section);
                                        setDrawerOpen(false);
                                    }}
                                >
                                    <Ionicons
                                        name={DRAWER_ICONS[section]}
                                        size={22}
                                        color={active ? "#1357a9" : "#333"}
                                    />
                                    <Text
                                        style={[styles.drawerItemLabel, active && styles.drawerItemLabelActive]}
                                    >
                                        {SECTION_LABELS[section]}
                                    </Text>
                                </Pressable>
                            );
                        })}
                    </SafeAreaView>
                </View>
            ) : null}
        </View>
    );
}

function FrameResultCard({ item }: { item: ImageResult }) {
    return (
        <View style={styles.frameCard}>
            <Text style={styles.frameTimestamp}>Timestamp: {formatTime(item.timestamp)}</Text>
            <Image source={{ uri: item.uri }} style={styles.frameImage} />
            <View style={styles.frameLabels}>
                {item.nsfw.map((n, i) => (
                    <Text key={n.label + i} style={styles.frameLabelLine}>
                        {n.label} ({(n.score * 100).toFixed(1)}%)
                    </Text>
                ))}
            </View>
        </View>
    );
}


const styles = StyleSheet.create({
    screenRoot: {
        flex: 1,
        backgroundColor: "#fff",
    },
    headerSafe: {
        backgroundColor: "#5f7591",
    },
    headerRow: {
        flexDirection: "row",
        alignItems: "center",
        paddingHorizontal: 8,
        paddingVertical: 10,
    },
    headerSideSlot: {
        width: 44,
        alignItems: "center",
        justifyContent: "center",
    },
    headerTitleText: {
        flex: 1,
        textAlign: "center",
        color: "#fff",
        fontSize: 18,
        fontWeight: "600",
    },
    bodyScroll: {
        flexGrow: 1,
        paddingVertical: 12,
        paddingHorizontal: 16,
        alignItems: "center",
    },
    sectionBlock: {
        width: "100%",
        alignItems: "stretch",
    },
    audioControlsRow: {
        flexDirection: "row",
        flexWrap: "wrap",
        justifyContent: "center",
        gap: 8,
        marginBottom: 16,
    },
    audioMeta: {
        width: "100%",
        padding: 12,
        backgroundColor: "#f0f4f8",
        borderRadius: 8,
    },
    metaLine: {
        fontSize: 15,
        color: "#333",
        marginBottom: 4,
    },
    mutedCenter: {
        marginTop: 24,
        fontSize: 16,
        color: "#666",
        textAlign: "center",
    },
    frameCard: {
        marginBottom: 20,
        alignItems: "center",
        width: "100%",
    },
    frameTimestamp: {
        fontSize: 15,
        fontWeight: "600",
        marginBottom: 8,
        color: "#222",
    },
    frameImage: {
        width: "100%",
        maxWidth: 320,
        aspectRatio: 16 / 10,
        borderRadius: 8,
        backgroundColor: "#eee",
    },
    frameLabels: {
        marginTop: 10,
        alignSelf: "stretch",
        maxWidth: 320,
    },
    frameLabelLine: {
        fontSize: 14,
        color: "#333",
        marginBottom: 4,
    },
    transcriptText: {
        fontSize: 16,
        lineHeight: 24,
        color: "#222",
        width: "100%",
    },
    moderationBlock: {
        marginTop: 24,
        width: "100%",
    },
    moderationHeading: {
        fontWeight: "700",
        fontSize: 17,
        marginBottom: 10,
        color: "#222",
    },
    moderationCard: {
        backgroundColor: "#f0f0f0",
        padding: 15,
        borderRadius: 10,
        width: "100%",
    },
    moderationRow: {
        flexDirection: "row",
        justifyContent: "space-between",
        marginBottom: 6,
    },
    moderationLabel: {
        color: "#111",
        fontSize: 15,
    },
    moderationLabelFlagged: {
        color: "#c00",
        fontWeight: "700",
    },
    moderationScore: {
        fontSize: 15,
        color: "#111",
    },
    moderationScoreFlagged: {
        color: "#c00",
        fontWeight: "600",
    },
    drawerRoot: {
        ...StyleSheet.absoluteFillObject,
        zIndex: 100,
    },
    drawerBackdrop: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: "rgba(0,0,0,0.45)",
    },
    drawerSheet: {
        position: "absolute",
        left: 0,
        top: 0,
        bottom: 0,
        width: 288,
        backgroundColor: "#eef1f6",
        borderRightWidth: StyleSheet.hairlineWidth,
        borderRightColor: "#ccc",
        paddingHorizontal: 12,
        paddingTop: 8,
    },
    drawerHeading: {
        fontSize: 13,
        fontWeight: "700",
        color: "#555",
        textTransform: "uppercase",
        letterSpacing: 0.6,
        marginBottom: 12,
        marginTop: 4,
    },
    drawerItem: {
        flexDirection: "row",
        alignItems: "center",
        gap: 12,
        paddingVertical: 14,
        paddingHorizontal: 10,
        borderRadius: 10,
        marginBottom: 4,
    },
    drawerItemActive: {
        backgroundColor: "#d8e4f5",
    },
    drawerItemLabel: {
        fontSize: 16,
        color: "#222",
        flex: 1,
    },
    drawerItemLabelActive: {
        color: "#1357a9",
        fontWeight: "600",
    },
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
    overlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(0,0,0,0.5)',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 200,
    },
    video: {
        width: '100%',
        aspectRatio: 16 / 9,
        marginVertical: 12,
        borderRadius: 8,
        backgroundColor: "#000",
    },
});
