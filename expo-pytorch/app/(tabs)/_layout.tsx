import { Ionicons } from "@expo/vector-icons";
import { Tabs } from "expo-router";

export default function TabLayout() {
  return(
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: '#f53dff',
        headerStyle: {
          backgroundColor: '#5f7591',
        },
        headerShadowVisible: false,
        headerTintColor: '#fff',
        tabBarStyle: {
          backgroundColor: '#1357a9',
        },
      }}
    >

      <Tabs.Screen 
        name="index" 
        options={{ 
          title: 'Images',
          tabBarIcon: ({ color, focused }) => (
            <Ionicons name={focused ? 'home-sharp': 'home-outline'} color={color} size={24}/>
          ),
        }}
      />

      <Tabs.Screen 
        name="aiScreen" 
        options={{ 
          title: 'Model',
          tabBarIcon: ({ color, focused }) => (
            <Ionicons name={focused ? 'information-circle': 'information-circle-outline'} color={color} size={24}/>
          ),
        }}
      />
    </Tabs>
  );
}