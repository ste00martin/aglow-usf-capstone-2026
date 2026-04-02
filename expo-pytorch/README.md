# expo-pytorch

This app uses Expo Router and `react-native-executorch` for on-device model inference.

## Quick start

Run these commands from `expo-pytorch/`:

1. Set up the app for your target platform:

   ```bash
   cd expo-pytorch
   npm run setup
   ```

   For iOS development, run this instead:

   ```bash
   npm run setup:ios
   ```

2. Start the native app:

   ```bash
   npm run ios
   ```

   Or for Android:

   ```bash
   npm run android
   ```

## Notes

- `npm run setup` installs JS dependencies and exports the shared XNNPACK `*.pte` assets used on Android and other non-iOS targets.
- `npm run setup:ios` does the same and also exports the iOS CoreML `*_coreml.pte` assets.
- `npm run ios` is the intended local run command for this app.
- `react-native-executorch` requires a native development build. Do not use Expo Go for this project.
- CoreML export requires macOS.
- The setup script creates `../.venv-executorch-export`, which is local-only and ignored by git.
- First-time export needs network access once to pull `blazeface.py`, `blazeface.pth`, and `anchors.npy` from [BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch). The setup/export commands fetch them automatically.

## Advanced commands

If you want the lower-level steps explicitly:

```bash
npm run models:setup
npm run models:export
npm run models:export:ios
npm run models:export:all
```

Those commands call the repo-root shell scripts in `../scripts/`. The export scripts automatically bootstrap the Python environment if it does not exist yet.
