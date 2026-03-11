import { Stack } from "expo-router";
import { AlbumContext } from "../AlbumContext";
import * as MediaLibrary from "expo-media-library";
import { useState } from "react";


export default function RootLayout() {
  const [assets, setAssets] = useState<MediaLibrary.Asset[]>([])
  return(
    <AlbumContext.Provider value={{ assets, setAssets}}>
      <Stack>
        <Stack.Screen name="(tabs)" options ={{headerShown:false}}/>
      </Stack>
    </AlbumContext.Provider>
  );
}