import { createContext } from "react";
import * as MediaLibrary from "expo-media-library";

// Context type
type AlbumContextType = {
  assets: MediaLibrary.Asset[];
  setAssets: (assets: MediaLibrary.Asset[]) => void;
};

// Create context with defaults
export const AlbumContext = createContext<AlbumContextType>({
  assets: [],
  setAssets: () => {},
});