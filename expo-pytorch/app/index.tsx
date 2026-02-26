import { useEffect, useState } from "react";
import { Alert, Pressable, Image, ScrollView, Text, View, Linking, StyleSheet } from "react-native";

import * as MediaLibrary from "expo-media-library";

export default function HomeScreen() {
  const [permissionResponse, requestPermission] = MediaLibrary.usePermissions();
  const [photos, setPhotos] = useState<MediaLibrary.Asset[]>([])

  useEffect(
    () => {
      const checkPermsAndLoadImages = async () =>{
        if(permissionResponse?.status === 'granted'){
          const permissionSpecific = await MediaLibrary.getPermissionsAsync();
          if ((permissionSpecific.accessPrivileges === 'all' || permissionSpecific.accessPrivileges === 'limited')){ // for distinguishing between limited and full library access
            await loadImages()
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
  
  const loadImages = async () => {
    const media = await MediaLibrary.getAssetsAsync(
      {
        mediaType: 'photo',
        sortBy: [['creationTime', true]],
      }
    );
    setPhotos(media.assets);
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

            {photos.map(photo => (
              <Image 
                key={photo.id} 
                source={{uri: photo.uri}}
                style={{
                  width: '50%',
                  aspectRatio: 1,
                  margin: 4,
                  borderRadius: 8,
                }}
              />
            ))}
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
});

