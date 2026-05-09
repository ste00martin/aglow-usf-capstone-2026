import { createContext, type Dispatch, type SetStateAction } from "react";

/** Assets chosen via image picker (and shared with Process Images). */
export type AlbumAsset = {
  id: string;
  uri: string;
  mediaType: "photo" | "video";
  /** Display URI when different from `uri` (e.g. video frame thumbnail). */
  displayUri?: string;
  /** Original dimensions when known (used by face crop on Process Images). */
  width?: number;
  height?: number;
};

type AlbumContextType = {
  assets: AlbumAsset[];
  setAssets: Dispatch<SetStateAction<AlbumAsset[]>>;
};

export const AlbumContext = createContext<AlbumContextType>({
  assets: [],
  setAssets: () => {},
});
