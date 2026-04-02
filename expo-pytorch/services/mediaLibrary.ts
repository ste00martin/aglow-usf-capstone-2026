import * as ImageManipulator from "expo-image-manipulator";
import * as MediaLibrary from "expo-media-library";

const LIBRARY_PAGE_SIZE = 120;

export async function getLocalUri(asset: MediaLibrary.Asset): Promise<string> {
  const info = await MediaLibrary.getAssetInfoAsync(asset.id);

  if (!info.localUri) {
    throw new Error(`Cannot get local URI for asset ${asset.id}`);
  }

  let localUri = info.localUri;

  if (localUri.startsWith("file://") && !localUri.toLowerCase().endsWith(".heic")) {
    return localUri;
  }

  if (localUri.toLowerCase().endsWith(".heic")) {
    const result = await ImageManipulator.manipulateAsync(
      localUri,
      [],
      { format: ImageManipulator.SaveFormat.JPEG, compress: 1 },
    );
    localUri = result.uri;
  }

  return localUri;
}

export async function loadLibraryAssets(limit = LIBRARY_PAGE_SIZE): Promise<MediaLibrary.Asset[]> {
  const collectedAssets: MediaLibrary.Asset[] = [];
  let after: string | undefined;

  while (collectedAssets.length < limit) {
    const page = await MediaLibrary.getAssetsAsync({
      after,
      first: Math.min(LIBRARY_PAGE_SIZE, limit - collectedAssets.length),
      mediaType: "photo",
      sortBy: [["creationTime", true]],
    });

    collectedAssets.push(...page.assets);

    if (!page.hasNextPage || !page.endCursor) {
      break;
    }

    after = page.endCursor;
  }

  const localUris = await Promise.all(collectedAssets.map((asset) => getLocalUri(asset)));

  return collectedAssets.map((asset, index) => ({
    ...asset,
    uri: localUris[index],
  }));
}
