# Quick Start
- Install npm and npx from the [Node.js](https://nodejs.org/en) official website.
- To run the app on an iOS Simulator, you must have a computer running MacOS. You will need to install Xcode, and then install the command line tools by going to Settings, then clicking Locations and selecting the most recent version from the Command Line Tools dropdown. Finally, to install the iOS simulator, open Xcode > Settings... > Components, and under Platform Support > iOS ..., click Get.

After cloning the app:
1.) Run the commands 'scripts/export_models_ios_coreml.sh' and 'scripts/export_models_xnnpack.sh' to get all of the models on your local machine
2.) Run the command 'cd expo-pytorch'
3.) In the 'expo-pytorch' folder, run 'npm install', then 'npx expo run:ios'

You now are running the app on the simulator. To add photos and videos for testing, click the home button at the top right, then add them in the Photos app directly from your computer. 


# Models Used
- [BlazeFace](https://github.com/hollance/BlazeFace-PyTorch) - facial recognition and cropping
- HuggingFace: [nateraw/vit-age-classifier](https://huggingface.co/nateraw/vit-age-classifier) - age estimation
- HuggingFace: [rizvandwiki/gender-classification](https://huggingface.co/rizvandwiki/gender-classification) - gender estimation
- HuggingFace: [Ateeqq/nsfw-image-detection](https://huggingface.co/Ateeqq/nsfw-image-detection) - NSFW image detection/classification
- HuggingFace: [ifmain/ModerationBERT-En-02](https://huggingface.co/ifmain/ModerationBERT-En-02) - NSFW text detection/classification
- Open source: [expo-video-audio-extractor](https://github.com/elliotfleming/expo-video-audio-extractor) - stripping audio track from a video
- Built-in to React Native Executorch: [useSpeechToText](https://docs.swmansion.com/react-native-executorch/docs/hooks/natural-language-processing/useSpeechToText) - converting the audio file into text

# Time Benchmarking for BlazeFace and HuggingFace models

The following were tested on [UTKFace](https://susanqq.github.io/UTKFace/) - dataset of over 24,000 faces of different ages
- BlazeFace:  
P50: 15.754 ms
P90: 21.670 ms
P99: 60.105 ms

- Age + gender:
P50: 106.449 ms
P90: 134.568 ms
P99: 236.987 ms

The Ateeqq/nsfw-image-detection model was tested on the [DETECTOR_AUTO_GENERATED_DATA](https://github.com/notAI-tech/NudeNet/releases/download/v0/DETECTOR_AUTO_GENERATED_DATA.zip) dataset, which consists of 20,000 NSFW images
- NSFW images:
P50: 117.319 ms
P90: 153.851 ms
P99: 290.513 ms

The ifmain/ModerationBERT-En-02 model was tested on 20,000 comments from the [Jigsaw Toxic Comment Dataset](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge/viewer/default/train?row=5)
- NSFW text:
P50 : 24.377 ms
P90 : 59.919 ms
P99 : 135.954 ms


