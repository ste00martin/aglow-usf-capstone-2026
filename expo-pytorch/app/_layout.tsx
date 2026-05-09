import { Stack } from "expo-router";
import { AlbumContext, type AlbumAsset } from "../AlbumContext";
import { useState } from "react";


export default function RootLayout() {
  const [assets, setAssets] = useState<AlbumAsset[]>([])
  return(
    <AlbumContext.Provider value={{ assets, setAssets}}>
      <Stack>
        <Stack.Screen name="(tabs)" options ={{headerShown:false}}/>
      </Stack>
    </AlbumContext.Provider>
  );
}