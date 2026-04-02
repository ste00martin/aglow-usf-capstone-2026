# expo-pytorch

This app uses Expo Router and `react-native-executorch` for on-device model inference.

## Fresh clone setup

Run these commands from the repo root:

1. Install the Expo app dependencies:

   ```bash
   cd expo-pytorch
   npm install
   cd ..
   ```

2. Create the local Python environment used to export the ExecuTorch models:

   ```bash
   ./scripts/setup_executorch_export_env.sh
   ```

3. Export the standard XNNPACK model assets:

   ```bash
   ./scripts/export_models_xnnpack.sh
   ```

4. If you are running on iOS, also export the CoreML model assets:

   ```bash
   ./scripts/export_models_ios_coreml.sh
   ```

5. Start the iOS app:

   ```bash
   cd expo-pytorch
   npm run ios
   ```

## Notes

- `npm run ios` is the intended local run command for this app.
- `react-native-executorch` requires a native development build. Do not use Expo Go for this project.
- CoreML export requires macOS.
- The setup script creates `../.venv-executorch-export`, which is local-only and ignored by git.
- `export_models_xnnpack.sh` writes the shared `*.pte` assets used on Android and other non-iOS targets.
- `export_models_ios_coreml.sh` writes parallel `*_coreml.pte` assets, and the app selects those on iOS.

## Alternative run commands

For Android:

```bash
cd expo-pytorch
npm run android
```
