import { useEffect, useMemo, useState } from "react";
import { Alert, Pressable, ScrollView, Text, View, StyleSheet, ActivityIndicator, useWindowDimensions } from "react-native";
import { Image } from 'expo-image';
import { Ionicons } from "@expo/vector-icons";
import { SafeAreaView, useSafeAreaInsets } from "react-native-safe-area-context";
import { getDefaultHeaderHeight } from "@react-navigation/elements";
import * as ImagePicker from 'expo-image-picker';
import { useVideoPlayer, VideoView } from 'expo-video';
import * as VideoThumbnails from 'expo-video-thumbnails';
import { useExecutorchModule, ScalarType, useSpeechToText, WHISPER_TINY } from "react-native-executorch";
import { extractAudio } from 'expo-video-audio-extractor';
import { AudioContext } from 'react-native-audio-api';
import { useAudioPlayer, useAudioPlayerStatus } from 'expo-audio';
import * as FileSystem from 'expo-file-system/legacy';
import type { TensorPtr } from "react-native-executorch";
import { imageUriToViTTensor, allFromLogits } from "../../aipreprocessing"
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

    const insets = useSafeAreaInsets();
    const window = useWindowDimensions();
    const headerTotalHeight = getDefaultHeaderHeight(
        { width: window.width, height: window.height },
        false,
        insets.top
    );

    useEffect(() => {
        if (!isReady || !running) return;
        const pickVideo = async () => {
            setThumbnails([]); // reset thumbnails when picking a new video
            setFlagged([]); // reset flagged results when picking a new video
            setTextModeration([]); // reset text moderation results
            setVideo(""); // reset video URI
            setAudio(null); // reset audio URI
            setTranscript(""); // reset transcript
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
                let start = Date.now()
                const transcribedAudio = await transcribeAudio(outputUri);
                console.log("Transcription took ", Date.now() - start, "ms")
                setTranscript(transcribedAudio)

                if (transcribedAudio) {
                    console.log("Transcribed Audio:", transcribedAudio);
                    try {
                        // We utilize standard BERT multilingual tokenization for ModerationBERT.
                        const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-multilingual-cased');
                        const tokens = await tokenizer(transcribedAudio, {
                            padding: 'max_length',
                            truncation: true,
                            max_length: 128,
                        });

                        const inputIdsData = tokens.input_ids.data instanceof Int32Array
                            ? tokens.input_ids.data
                            : new Int32Array(Array.from(tokens.input_ids.data, x => Number(x)));

                        const attentionMaskData = tokens.attention_mask.data instanceof Int32Array
                            ? tokens.attention_mask.data
                            : new Int32Array(Array.from(tokens.attention_mask.data, x => Number(x)));

                        const inputIdsPtr: TensorPtr = {
                            dataPtr: inputIdsData,
                            sizes: [1, 128],
                            scalarType: ScalarType.INT,
                        };

                        const attentionMaskPtr: TensorPtr = {
                            dataPtr: attentionMaskData,
                            sizes: [1, 128],
                            scalarType: ScalarType.INT,
                        };

                        console.log("Pre-flight Token Shape Validation:", inputIdsData.length, attentionMaskData.length);
                        console.log("Sample Tokens:", Array.from(inputIdsData).slice(0, 10));

                        const textOutputs = await textModel.forward([inputIdsPtr, attentionMaskPtr]);
                        //const textLogits = new Float32Array(textOutputs[0].dataPtr as ArrayBuffer);
                        const rawData = textOutputs[0].dataPtr;
                        const textLogits = rawData instanceof Float32Array
                            ? rawData
                            : rawData instanceof ArrayBuffer
                                ? new Float32Array(rawData)
                                : Float32Array.from(rawData as unknown as number[]);
                        console.log("textLogits: ", textLogits.length, Array.from(textLogits));
                        const textLabels = [
                            'Harassment', 'Harassment (Threatening)', 'Hate Speech', 'Hate Speech (Threatening)',
                            'Self Harm', 'Self Harm (Instructions)', 'Self Harm (Intent)', 'Explicit (Sexual)',
                            'Explicit (Minors)', 'Violence', 'Violence (Graphic)', 'Self Harm ',
                            'Explicit (Minors) ', 'Hate Speech (Threatening) ', 'Violence (Graphic) ',
                            'Self Harm (Intent) ', 'Self Harm (Instructions) ', 'Harassment (Threatening) '
                        ];
                        const textResultRaw = Array.from(textLogits, (logit, idx) => ({
                            label: textLabels[idx] ?? `class_${idx}`,
                            score: 1 / (1 + Math.exp(-logit))
                        }));

                        const uniqueResultMap = new Map<string, number>();
                        for (const r of textResultRaw) {
                            const cleanLabel = r.label.trim();
                            if (!uniqueResultMap.has(cleanLabel) || uniqueResultMap.get(cleanLabel)! < r.score) {
                                uniqueResultMap.set(cleanLabel, r.score);
                            }
                        }
                        const textResult = Array.from(uniqueResultMap.entries()).map(([label, score]) => ({ label, score }));

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
        console.log("Audio URI:", audioUri);
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
            <View style={styles.loadingScreen}>
                <ActivityIndicator size="large" color="#00c8ff" />
                <Text style={styles.loadingSubtitle}>Loading models...</Text>
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
                                    accessibilityRole="button"
                                    accessibilityLabel="Pause"
                                    style={styles.buttonContainer}
                                    onPress={() => {
                                        audioPlayer.pause();
                                        setAudioButtonStatus(false);
                                    }}
                                >
                                    <Ionicons name="pause-outline" size={24} color="#fff" />
                                </Pressable>
                            ) : (
                                <Pressable
                                    accessibilityRole="button"
                                    accessibilityLabel="Play"
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
                                    <Ionicons name="play-outline" size={24} color="#fff" />
                                </Pressable>
                            )}
                            <Pressable
                                accessibilityRole="button"
                                accessibilityLabel="Replay"
                                style={styles.buttonContainer}
                                onPress={() => {
                                    if (!audioPlayer) return;
                                    audioPlayer.seekTo(0);
                                    audioPlayer.play();
                                    setAudioButtonStatus(true);
                                }}
                            >
                                <Ionicons name="reload-outline" size={24} color="#fff" />
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
            case "transcript": {
                const nonZeroModeration = textModeration.filter(
                    (res) => (res.score * 100).toFixed(1) !== "0.0"
                );
                return (
                    <View style={styles.sectionBlock}>
                        <Text style={styles.transcriptText}>
                            {transcript || "No transcription yet…"}
                        </Text>
                        {textModeration.length > 0 && (
                            <View style={styles.moderationBlock}>
                                <Text style={styles.moderationHeading}>Text moderation</Text>
                                <View style={styles.moderationCard}>
                                    {nonZeroModeration.length > 0 ? (
                                        nonZeroModeration.map((res, i) => (
                                            <View key={res.label + i} style={styles.moderationRow}>
                                                <Text
                                                    style={[
                                                        styles.moderationLabel,
                                                        res.score > 0.4 && styles.moderationLabelFlagged,
                                                    ]}
                                                >
                                                    {res.label}
                                                </Text>
                                                <Text
                                                    style={[
                                                        styles.moderationScore,
                                                        res.score > 0.4 && styles.moderationScoreFlagged,
                                                    ]}
                                                >
                                                    {(res.score * 100).toFixed(1)}%
                                                </Text>
                                            </View>
                                        ))
                                    ) : (
                                        <Text style={styles.moderationEmpty}>
                                            All toxicity categories at 0.0%
                                        </Text>
                                    )}
                                </View>
                            </View>
                        )}
                    </View>
                );
            }
            default:
                return null;
        }
    };

    return (
        <View style={styles.screenRoot}>
            {running && (
                <View style={styles.overlay}>
                    <ActivityIndicator size="large" color="#00c8ff" />
                </View>
            )}
            <View
                style={[
                    styles.headerSafe,
                    { height: headerTotalHeight, paddingTop: insets.top },
                ]}
            >
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
            </View>

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
                                        color={active ? "#00c8ff" : "#555"}
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
    loadingScreen: {
        flex: 1,
        justifyContent: "center",
        alignItems: "center",
        backgroundColor: "#ffffff",
    },
    loadingSubtitle: {
        marginTop: 10,
        fontSize: 16,
        color: "#333",
    },
    screenRoot: {
        flex: 1,
        backgroundColor: "#ffffff",
    },
    headerSafe: {
        backgroundColor: "#000000a7",
    },
    headerRow: {
        flex: 1,
        flexDirection: "row",
        alignItems: "center",
        paddingHorizontal: 8,
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
        fontSize: 17,
        fontWeight: "600",
    },
    bodyScroll: {
        flexGrow: 1,
        paddingVertical: 12,
        paddingHorizontal: 16,
        alignItems: "center",
        backgroundColor: "#ffffff",
    },
    sectionBlock: {
        width: "100%",
        alignItems: "stretch",
    },
    audioControlsRow: {
        flexDirection: "row",
        flexWrap: "wrap",
        justifyContent: "center",
        marginBottom: 16,
    },
    audioMeta: {
        width: "100%",
        padding: 12,
        backgroundColor: "#f5f5f5",
        borderRadius: 8,
        borderWidth: StyleSheet.hairlineWidth,
        borderColor: "#e0e0e0",
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
        color: "#111",
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
        color: "#111",
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
        color: "#111",
    },
    moderationCard: {
        backgroundColor: "#f5f5f5",
        padding: 15,
        borderRadius: 10,
        width: "100%",
        borderWidth: StyleSheet.hairlineWidth,
        borderColor: "#e0e0e0",
    },
    moderationRow: {
        flexDirection: "row",
        justifyContent: "space-between",
        marginBottom: 6,
    },
    moderationLabel: {
        color: "#222",
        fontSize: 15,
        flex: 1,
        marginRight: 8,
    },
    moderationLabelFlagged: {
        color: "#c0392b",
        fontWeight: "700",
    },
    moderationScore: {
        fontSize: 15,
        color: "#222",
    },
    moderationScoreFlagged: {
        color: "#c0392b",
        fontWeight: "600",
    },
    moderationEmpty: {
        textAlign: "center",
        fontStyle: "italic",
        color: "#666",
    },
    drawerRoot: {
        ...StyleSheet.absoluteFillObject,
        zIndex: 100,
    },
    drawerBackdrop: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: "rgba(0,0,0,0.35)",
    },
    drawerSheet: {
        position: "absolute",
        left: 0,
        top: 0,
        bottom: 0,
        width: 288,
        backgroundColor: "#ffffff",
        borderRightWidth: StyleSheet.hairlineWidth,
        borderRightColor: "#e0e0e0",
        paddingHorizontal: 12,
        paddingTop: 8,
    },
    drawerHeading: {
        fontSize: 13,
        fontWeight: "700",
        color: "#666",
        textTransform: "uppercase",
        letterSpacing: 0.6,
        marginBottom: 12,
        marginTop: 4,
    },
    drawerItem: {
        flexDirection: "row",
        alignItems: "center",
        paddingVertical: 14,
        paddingHorizontal: 10,
        borderRadius: 10,
        marginBottom: 4,
    },
    drawerItemActive: {
        backgroundColor: "rgba(0, 200, 255, 0.12)",
    },
    drawerItemLabel: {
        fontSize: 16,
        color: "#222",
        flex: 1,
        marginLeft: 12,
    },
    drawerItemLabelActive: {
        color: "#00c8ff",
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
        backgroundColor: 'rgba(0,0,0,0.45)',
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
