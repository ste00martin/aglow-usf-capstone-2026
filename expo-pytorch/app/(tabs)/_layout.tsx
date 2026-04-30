import { Ionicons } from "@expo/vector-icons";
import { Tabs } from "expo-router";

export default function TabLayout() {
  return(
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: '#00c8ff',
        headerStyle: {
          backgroundColor: '#000000a7',
        },
        headerShadowVisible: false,
        headerTintColor: '#fff',
        tabBarStyle: {
          backgroundColor: '#000000da',
        },
      }}
    >

      <Tabs.Screen
        name="index"
        options={{
          title: 'Images',
          tabBarIcon: ({ color, focused }) => (
            <Ionicons name={focused ? 'images': 'images-outline'} color={color} size={24}/>
          ),
        }}
      />

      <Tabs.Screen
        name="feed"
        options={{
          title: 'Feed',
          tabBarIcon: ({ color, focused }) => (
            <Ionicons name={focused ? 'play-circle' : 'play-circle-outline'} color={color} size={24} />
          ),
        }}
      />

      <Tabs.Screen 
        name="aiScreen" 
        options={{ 
          title: 'Process Images',
          tabBarIcon: ({ color, focused }) => (
            <Ionicons name={focused ? 'information-circle': 'information-circle-outline'} color={color} size={24}/>
          ),
        }}
      />

      <Tabs.Screen 
        name="videoupload" 
        options={{ 
          title: 'Video Upload',
          tabBarIcon: ({ color, focused }) => (
            <Ionicons name={focused ? 'videocam': 'videocam-outline'} color={color} size={24}/>
          ),
        }}
      />
    </Tabs>

    
  );
}