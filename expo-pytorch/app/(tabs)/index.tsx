import { useEffect, useState } from "react";
import { Alert, Pressable, Image, ScrollView, Text, View, Linking, StyleSheet, TouchableOpacity } from "react-native";
import { AlbumContext } from "../../AlbumContext";
import { useContext } from "react";

import * as MediaLibrary from 'expo-media-library';
import * as ImageManipulator from 'expo-image-manipulator';
import Ionicons from "@expo/vector-icons/build/Ionicons";
import * as VideoThumbnails from 'expo-video-thumbnails';


export async function getLocalUri(asset: MediaLibrary.Asset): Promise<string> {
  // 1. Get the local file path for the asset
  const info = await MediaLibrary.getAssetInfoAsync(asset.id);
  if (!info.localUri) throw new Error(`Cannot get local URI for asset ${asset.id}`);

  const isVideo = asset.mediaType === 'video'; // filter out videos
  let localUri = info.localUri;
  if (isVideo) {
    //console.log("Major Debug:", localUri)
    return localUri;
  }

  // 2. Early return if already a local file and not HEIC
  if (localUri.startsWith('file://') && !localUri.toLowerCase().endsWith('.heic')) {
    return localUri;
  }

  // 3. If HEIC, convert to JPEG using ImageManipulator
  if (localUri.toLowerCase().endsWith('.heic')) {
    console.log("Detected HEIC:", localUri);
    const result = await ImageManipulator.manipulateAsync(
      localUri,
      [], // no resize or rotation
      { format: ImageManipulator.SaveFormat.JPEG, compress: 1 }
    );
    localUri = result.uri;
  }

  // 4. Return the local URI (guaranteed to be a file:// path)
  return localUri;
}



export default function HomeScreen() {
  const [permissionResponse, requestPermission] = MediaLibrary.usePermissions();
  //const [photos, setPhotos] = useState<MediaLibrary.Asset[]>([])
  const { assets, setAssets } = useContext(AlbumContext);

  useEffect(
    () => {
      const checkPermsAndLoadImages = async () =>{
        if(permissionResponse?.status === 'granted'){
          const permissionSpecific = await MediaLibrary.getPermissionsAsync();
          if ((permissionSpecific.accessPrivileges === 'all' || permissionSpecific.accessPrivileges === 'limited')){ // for distinguishing between limited and full library access
            await loadMedia()
          }
        }
        else if(permissionResponse?.status === 'denied' && permissionResponse.canAskAgain === false){
            Alert.alert(
                "Permission Required",
                "Photo access is disabled. Please enable it in Settings.",
                [
                    {text: "Cancel", style: "cancel"},
                    {text: "Open Settings", onPress: () => Linking.openSettings()},
                ],
                { cancelable: true }
            );
        }
      };

      checkPermsAndLoadImages()
    }, [permissionResponse] // called upon permissionResponse being called
  );
  
  const loadMedia = async () => {
    const media = await MediaLibrary.getAssetsAsync({
      mediaType: ['photo', 'video'],
      sortBy: [['creationTime', true]],
    });

    const convertedUris = await Promise.all(
      media.assets.map(async (asset) => {
        const uri = await getLocalUri(asset); // existing function
        // If video, generate thumbnail
        if (asset.mediaType === 'video') {
          try {
            const { uri: thumbnail } = await VideoThumbnails.getThumbnailAsync(uri, { time: 1000 });
            return thumbnail;
          } 
          catch (e) {
            console.warn('Failed to generate thumbnail', e);
            return uri; // fallback to video URI
          }
        }
        return uri; // photo
      })
    );
    const updatedAssets: (MediaLibrary.Asset & { displayUri: string })[] = []; // store both the original asset and the thumbnail.
    
    for (let i = 0; i < media.assets.length; i++) {
      const asset = media.assets[i];
      updatedAssets.push({
        ...asset, // create a new object
        displayUri: convertedUris[i], // override URI
      });
    }
    setAssets(updatedAssets); // spread to create a new array reference
  };
  

  const changeAccess = async () => {
    if(permissionResponse?.status === 'granted'){
          const permissionSpecific = await MediaLibrary.getPermissionsAsync();
          if ((permissionSpecific.accessPrivileges === 'all')){ // for distinguishing between limited and full library access
            console.log("Failed to change permissions, already set to 'None. Change Access in settings")
            Alert.alert(
                "Full Libary Access is already enabled.",
                "To change access, go to Settings.",
                [
                    {text: "Cancel", style: "cancel"},
                    {text: "Open Settings", onPress: () => Linking.openSettings()},
                ],
                { cancelable: true }
            );
        }
        else if ((permissionSpecific.accessPrivileges === 'limited')){ // for distinguishing between limited and full library access
            console.log("Failed to change permissions, already set to 'Entire Library. Change Access in settings")
            await MediaLibrary.presentPermissionsPickerAsync();
        }
    }
    else{
      await MediaLibrary.presentPermissionsPickerAsync();
      console.log("Changing permissions...")
    }
  }
  const openAppSettings = () => {
    Linking.openSettings(); // opens the app’s settings page
  };


  return (
    <ScrollView contentContainerStyle={{padding: 16, alignItems: 'center'}}>
      <Text style={{fontSize: 18, fontWeight: 'bold'}}>
        some images below
      </Text>
      
      {permissionResponse?.status !== 'granted'? (
        <Pressable style={styles.buttonContainer} onPress={requestPermission}>
              <Text style={styles.button}>Get Started (maybe?)</Text>
        </Pressable>
      ) : (
        <>
          <View style={{flexDirection: 'column', alignItems: 'center'}}>
            <Pressable style={styles.buttonContainer} onPress={changeAccess}>
              <Text style={styles.button}>Pick more photos.</Text>
            </Pressable>
            
            <Pressable style={styles.buttonContainer} onPress={openAppSettings}>
              <Text style={styles.button}>Change Access.</Text>
            </Pressable>

            {assets.map(asset => {
              if (asset.mediaType === 'photo') {
                return (
                  <Image
                    key={asset.id}
                    source={{ uri: asset.uri }}
                    style={styles.mediaItem}
                  />
                );
              } else if(asset.mediaType === 'video') {
                return(
                <TouchableOpacity key={asset.id} onPress={() => {}}>
                    <Image
                      source={{ uri: asset.uri }} // thumbnail
                      style={styles.mediaItem}
                    />
                    <Ionicons
                      name="play-circle-outline"
                      size={48}
                      color="white"
                      style={styles.playIcon}
                    />
                </TouchableOpacity>
                );
              }
            })}
          </View>
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({

    button:{
      fontSize: 20,
      color: '#fff',
    },
    buttonContainer: {
      paddingVertical: 10,
      paddingHorizontal: 16,
      borderRadius: 8,
      backgroundColor: '#333',
      margin: 8,
      alignItems: 'center',
    },
    mediaItem: {
      width: 150,          // fixed width or percentage
      height: 150,         // keeps square
      margin: 4,
      borderRadius: 8,
    },
    playIcon: {
      position: 'absolute',
      top: 50,
      left: 50,
    },
});

