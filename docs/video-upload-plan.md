# Video Upload Plan

## What exists now

The current Expo app is photo-only and on-device:

- `expo-pytorch/services/mediaLibrary.ts` only loads `mediaType: "photo"`.
- `expo-pytorch/services/analysisPipeline.ts` assumes every asset is a still image and runs image NSFW plus face/demographic models locally.
- `expo-pytorch/hooks/useAnalysisJob.ts` only schedules image analysis results.
- The UI copy in the current `expo-pytorch/app/index.tsx` and `expo-pytorch/components/workflow/TrustPillRow.tsx` is centered on photos and "No uploads".

I did not find an existing in-repo plan or implementation for video upload. The older deleted `expo-pytorch/app/(tabs)/aiScreen.tsx` also only documented the photo pipeline.

## Recommended shape

Do not put OpenAI moderation calls directly in the Expo client. Keep API keys and file processing on a backend or trusted worker.

Recommended flow:

1. Let the app select video assets from the library.
2. Upload the selected video to a backend endpoint.
3. On the backend:
   - transcribe the audio track
   - run text moderation on the transcript
   - sample one frame every 6 seconds
   - run image moderation on each sampled frame
4. Return a moderation report to the app before any publish step.

## App changes needed

1. Update `expo-pytorch/services/mediaLibrary.ts` to load videos in addition to photos.
2. Split workflow types so an item can be either an image job or a video job.
3. Keep the current on-device photo pipeline for images.
4. Route videos to a backend moderation request instead of the local ExecuTorch image pipeline.
5. Extend result types to include:
   - transcript moderation result
   - sampled-frame moderation result
   - timestamps of flagged frames

## Backend contract

Suggested request:

```json
POST /api/moderate-video
{
  "assetId": "local-or-server-id",
  "filename": "clip.mp4"
}
```

Suggested response:

```json
{
  "assetId": "local-or-server-id",
  "transcript": {
    "flagged": false,
    "flaggedChunks": []
  },
  "frames": {
    "intervalSeconds": 6,
    "flagged": true,
    "flaggedFrames": [
      {
        "timestampSecondsApprox": 24,
        "categories": ["sexual"]
      }
    ]
  },
  "shouldBlock": true
}
```

## Immediate next step

Use `scripts/moderate_video.js` as the first backend-side worker. It handles transcript moderation plus frame moderation every 6 seconds and can be wrapped by an API route later.
