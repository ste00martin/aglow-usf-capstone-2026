#!/usr/bin/env node

const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const process = require("node:process");
const { spawn } = require("node:child_process");

const DEFAULT_FRAME_INTERVAL_SECONDS = 6;
const DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe";
const DEFAULT_MODERATION_MODEL = "omni-moderation-latest";
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL ?? "https://api.openai.com";
const TRANSCRIPTION_READY_EXTENSIONS = new Set([
  ".flac",
  ".m4a",
  ".mp3",
  ".mp4",
  ".mpeg",
  ".mpga",
  ".ogg",
  ".wav",
  ".webm",
]);

function printUsage() {
  console.log(`
Usage:
  OPENAI_API_KEY=... node scripts/moderate_video.js --video path/to/video.mp4
  OPENAI_API_KEY=... node scripts/moderate_video.js --audio path/to/audio.m4a
  OPENAI_API_KEY=... node scripts/moderate_video.js --transcript-file path/to/transcript.txt

Options:
  --video PATH                 Video file to transcribe and/or sample for frame moderation
  --audio PATH                 Audio file to transcribe
  --transcript-file PATH       Existing transcript text file
  --frame-interval SECONDS     Sampling interval for video frames (default: 6)
  --language CODE              Optional ISO-639-1 language hint, e.g. en
  --transcription-model MODEL  Default: gpt-4o-mini-transcribe
  --moderation-model MODEL     Default: omni-moderation-latest
  --skip-transcript            Skip transcription and transcript moderation
  --skip-frames                Skip video frame moderation
  --keep-frames                Keep extracted frame images on disk
  --output PATH                Save the JSON report to a file
  --help                       Show this help
`);
}

function parseArgs(argv) {
  const options = {
    audio: null,
    frameIntervalSeconds: DEFAULT_FRAME_INTERVAL_SECONDS,
    keepFrames: false,
    language: null,
    moderationModel: DEFAULT_MODERATION_MODEL,
    output: null,
    skipFrames: false,
    skipTranscript: false,
    transcriptionModel: DEFAULT_TRANSCRIPTION_MODEL,
    transcriptFile: null,
    video: null,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    switch (arg) {
      case "--audio":
        options.audio = argv[++index] ?? null;
        break;
      case "--frame-interval":
        options.frameIntervalSeconds = Number(argv[++index] ?? DEFAULT_FRAME_INTERVAL_SECONDS);
        break;
      case "--help":
        options.help = true;
        break;
      case "--keep-frames":
        options.keepFrames = true;
        break;
      case "--language":
        options.language = argv[++index] ?? null;
        break;
      case "--moderation-model":
        options.moderationModel = argv[++index] ?? DEFAULT_MODERATION_MODEL;
        break;
      case "--output":
        options.output = argv[++index] ?? null;
        break;
      case "--skip-frames":
        options.skipFrames = true;
        break;
      case "--skip-transcript":
        options.skipTranscript = true;
        break;
      case "--transcript-file":
        options.transcriptFile = argv[++index] ?? null;
        break;
      case "--transcription-model":
        options.transcriptionModel = argv[++index] ?? DEFAULT_TRANSCRIPTION_MODEL;
        break;
      case "--video":
        options.video = argv[++index] ?? null;
        break;
      default:
        throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return options;
}

function getRequiredApiKey() {
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is required.");
  }

  return apiKey;
}

function toAbsolutePath(filePath) {
  return path.resolve(process.cwd(), filePath);
}

async function assertFileExists(filePath, label) {
  try {
    await fs.access(filePath);
  } catch {
    throw new Error(`${label} does not exist: ${filePath}`);
  }
}

function guessMimeType(filePath) {
  switch (path.extname(filePath).toLowerCase()) {
    case ".flac":
      return "audio/flac";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".m4a":
      return "audio/mp4";
    case ".mp3":
      return "audio/mpeg";
    case ".mp4":
      return "video/mp4";
    case ".ogg":
      return "audio/ogg";
    case ".png":
      return "image/png";
    case ".wav":
      return "audio/wav";
    case ".webm":
      return "video/webm";
    default:
      return "application/octet-stream";
  }
}

async function fileFromPath(filePath) {
  const buffer = await fs.readFile(filePath);

  return new File([buffer], path.basename(filePath), {
    type: guessMimeType(filePath),
  });
}

async function openAiJson(endpoint, payload) {
  const response = await fetch(`${OPENAI_BASE_URL}${endpoint}`, {
    body: JSON.stringify(payload),
    headers: {
      Authorization: `Bearer ${getRequiredApiKey()}`,
      "Content-Type": "application/json",
    },
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(await buildOpenAiError(response));
  }

  return response.json();
}

async function openAiMultipart(endpoint, formData) {
  const response = await fetch(`${OPENAI_BASE_URL}${endpoint}`, {
    body: formData,
    headers: {
      Authorization: `Bearer ${getRequiredApiKey()}`,
    },
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(await buildOpenAiError(response));
  }

  return response.json();
}

async function buildOpenAiError(response) {
  const text = await response.text();

  try {
    const parsed = JSON.parse(text);
    const message = parsed.error?.message ?? text;

    return `OpenAI API error ${response.status}: ${message}`;
  } catch {
    return `OpenAI API error ${response.status}: ${text}`;
  }
}

function flaggedCategories(result) {
  return Object.entries(result?.categories ?? {})
    .filter(([, value]) => Boolean(value))
    .map(([key]) => key);
}

function topCategoryScores(result, limit = 5) {
  return Object.entries(result?.category_scores ?? {})
    .sort((left, right) => Number(right[1]) - Number(left[1]))
    .slice(0, limit)
    .map(([category, score]) => ({
      category,
      score: Number(score),
    }));
}

function chunkText(text, maxChars = 2000) {
  const trimmed = text.trim();

  if (!trimmed) {
    return [];
  }

  const words = trimmed.split(/\s+/);
  const chunks = [];
  let current = "";

  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;

    if (candidate.length > maxChars && current) {
      chunks.push(current);
      current = word;
      continue;
    }

    current = candidate;
  }

  if (current) {
    chunks.push(current);
  }

  return chunks;
}

async function transcribeFile(filePath, options) {
  const formData = new FormData();
  formData.set("file", await fileFromPath(filePath));
  formData.set("model", options.transcriptionModel);
  formData.set("response_format", "json");

  if (options.language) {
    formData.set("language", options.language);
  }

  return openAiMultipart("/v1/audio/transcriptions", formData);
}

async function moderateText(text, moderationModel) {
  const response = await openAiJson("/v1/moderations", {
    input: text,
    model: moderationModel,
  });

  const result = response.results?.[0] ?? {};

  return {
    categories: flaggedCategories(result),
    flagged: Boolean(result.flagged),
    raw: result,
    topScores: topCategoryScores(result),
  };
}

async function moderateImage(filePath, moderationModel) {
  const buffer = await fs.readFile(filePath);
  const dataUrl = `data:${guessMimeType(filePath)};base64,${buffer.toString("base64")}`;
  const response = await openAiJson("/v1/moderations", {
    input: [
      {
        image_url: {
          url: dataUrl,
        },
        type: "image_url",
      },
    ],
    model: moderationModel,
  });
  const result = response.results?.[0] ?? {};

  return {
    categories: flaggedCategories(result),
    flagged: Boolean(result.flagged),
    raw: result,
    topScores: topCategoryScores(result),
  };
}

function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.on("error", (error) => {
      reject(error);
    });
    child.on("close", (code) => {
      if (code === 0) {
        resolve({ stderr, stdout });
        return;
      }

      reject(new Error(`${command} exited with code ${code}${stderr ? `: ${stderr.trim()}` : ""}`));
    });
  });
}

async function commandExists(command) {
  try {
    await runCommand(command, ["-version"]);
    return true;
  } catch {
    return false;
  }
}

async function extractAudioToTemp(videoPath) {
  if (!(await commandExists("ffmpeg"))) {
    throw new Error(
      "ffmpeg is required to extract audio from non-supported video formats. Install ffmpeg or pass --audio/--transcript-file.",
    );
  }

  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "moderate-video-audio-"));
  const audioPath = path.join(tempDir, "audio.m4a");

  await runCommand("ffmpeg", [
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-i",
    videoPath,
    "-vn",
    "-acodec",
    "aac",
    audioPath,
  ]);

  return { audioPath, tempDir };
}

async function extractFrames(videoPath, intervalSeconds) {
  if (!(await commandExists("ffmpeg"))) {
    throw new Error("ffmpeg is required for frame sampling. Install ffmpeg or rerun with --skip-frames.");
  }

  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "moderate-video-frames-"));
  const outputPattern = path.join(tempDir, "frame-%05d.jpg");

  await runCommand("ffmpeg", [
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-i",
    videoPath,
    "-vf",
    `fps=1/${intervalSeconds}`,
    "-q:v",
    "3",
    outputPattern,
  ]);

  const entries = await fs.readdir(tempDir);
  const framePaths = entries
    .filter((entry) => entry.toLowerCase().endsWith(".jpg"))
    .sort()
    .map((entry) => path.join(tempDir, entry));

  return { framePaths, tempDir };
}

async function mapWithConcurrency(items, limit, worker) {
  const results = new Array(items.length);
  let nextIndex = 0;

  async function runner() {
    while (nextIndex < items.length) {
      const currentIndex = nextIndex;
      nextIndex += 1;
      results[currentIndex] = await worker(items[currentIndex], currentIndex);
    }
  }

  const count = Math.min(limit, items.length);
  await Promise.all(Array.from({ length: count }, runner));

  return results;
}

async function buildTranscriptReport(options) {
  if (options.skipTranscript) {
    return null;
  }

  let transcriptText = "";
  let transcriptSource = null;
  let extractedAudioTempDir = null;

  try {
    if (options.transcriptFile) {
      transcriptSource = {
        kind: "transcript_file",
        path: options.transcriptFile,
      };
      transcriptText = await fs.readFile(options.transcriptFile, "utf8");
    } else if (options.audio) {
      transcriptSource = {
        kind: "audio_file",
        path: options.audio,
      };
      const transcription = await transcribeFile(options.audio, options);
      transcriptText = transcription.text ?? "";
    } else if (options.video) {
      const videoExtension = path.extname(options.video).toLowerCase();

      if (TRANSCRIPTION_READY_EXTENSIONS.has(videoExtension)) {
        transcriptSource = {
          kind: "video_file",
          path: options.video,
        };
        const transcription = await transcribeFile(options.video, options);
        transcriptText = transcription.text ?? "";
      } else {
        const extracted = await extractAudioToTemp(options.video);

        extractedAudioTempDir = extracted.tempDir;
        transcriptSource = {
          kind: "extracted_audio",
          path: extracted.audioPath,
        };

        const transcription = await transcribeFile(extracted.audioPath, options);
        transcriptText = transcription.text ?? "";
      }
    } else {
      return null;
    }

    const chunks = chunkText(transcriptText);
    const moderatedChunks = [];

    for (let index = 0; index < chunks.length; index += 1) {
      const chunk = chunks[index];
      const moderation = await moderateText(chunk, options.moderationModel);

      moderatedChunks.push({
        categories: moderation.categories,
        chunkIndex: index,
        flagged: moderation.flagged,
        topScores: moderation.topScores,
      });
    }

    const flaggedChunks = moderatedChunks.filter((chunk) => chunk.flagged);

    return {
      chunkCount: chunks.length,
      flagged: flaggedChunks.length > 0,
      flaggedChunks,
      preview: transcriptText.slice(0, 280),
      source: transcriptSource,
      textLength: transcriptText.length,
    };
  } finally {
    if (extractedAudioTempDir) {
      await fs.rm(extractedAudioTempDir, { force: true, recursive: true });
    }
  }
}

async function buildFrameReport(options) {
  if (options.skipFrames || !options.video) {
    return null;
  }

  const extracted = await extractFrames(options.video, options.frameIntervalSeconds);

  try {
    const moderatedFrames = await mapWithConcurrency(
      extracted.framePaths,
      3,
      async (framePath, index) => {
        const moderation = await moderateImage(framePath, options.moderationModel);
        const result = {
          categories: moderation.categories,
          flagged: moderation.flagged,
          frameIndex: index,
          timestampSecondsApprox: index * options.frameIntervalSeconds,
          topScores: moderation.topScores,
        };

        if (options.keepFrames) {
          result.framePath = framePath;
        }

        return result;
      },
    );

    const flaggedFrames = moderatedFrames.filter((frame) => frame.flagged);

    return {
      flagged: flaggedFrames.length > 0,
      flaggedFrames,
      intervalSeconds: options.frameIntervalSeconds,
      keptFrameDirectory: options.keepFrames ? extracted.tempDir : null,
      totalFramesChecked: moderatedFrames.length,
    };
  } finally {
    if (!options.keepFrames) {
      await fs.rm(extracted.tempDir, { force: true, recursive: true });
    }
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));

  if (options.help) {
    printUsage();
    return;
  }

  if (!options.video && !options.audio && !options.transcriptFile) {
    throw new Error("Pass at least one of --video, --audio, or --transcript-file.");
  }

  if (options.skipTranscript && (options.skipFrames || !options.video)) {
    throw new Error("No moderation checks are enabled. Enable transcript checks or provide --video without --skip-frames.");
  }

  if (options.video) {
    options.video = toAbsolutePath(options.video);
    await assertFileExists(options.video, "Video file");
  }

  if (options.audio) {
    options.audio = toAbsolutePath(options.audio);
    await assertFileExists(options.audio, "Audio file");
  }

  if (options.transcriptFile) {
    options.transcriptFile = toAbsolutePath(options.transcriptFile);
    await assertFileExists(options.transcriptFile, "Transcript file");
  }

  if (!Number.isFinite(options.frameIntervalSeconds) || options.frameIntervalSeconds <= 0) {
    throw new Error("--frame-interval must be a positive number.");
  }

  getRequiredApiKey();

  const transcript = await buildTranscriptReport(options);
  const frames = await buildFrameReport(options);
  const shouldBlock = Boolean(transcript?.flagged || frames?.flagged);
  const report = {
    checkedAt: new Date().toISOString(),
    frames,
    moderationModel: options.moderationModel,
    shouldBlock,
    transcript,
    transcriptionModel: transcript ? options.transcriptionModel : null,
    video: options.video
      ? {
          path: options.video,
        }
      : null,
  };

  const serialized = `${JSON.stringify(report, null, 2)}\n`;

  if (options.output) {
    const outputPath = toAbsolutePath(options.output);
    await fs.writeFile(outputPath, serialized, "utf8");
    console.error(`Saved moderation report to ${outputPath}`);
  }

  process.stdout.write(serialized);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exitCode = 1;
});
