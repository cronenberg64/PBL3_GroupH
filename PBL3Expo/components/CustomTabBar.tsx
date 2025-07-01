import React from 'react';
import { View, TouchableOpacity, StyleSheet, Text } from 'react-native';
import { Home, Camera, Settings } from 'lucide-react-native';
import { useRouter, usePathname } from 'expo-router';

const CustomTabBar = () => {
  const router = useRouter();
  const pathname = usePathname();

  const tabs = [
    { name: 'Home', icon: Home, route: '/(tabs)' },
    { name: 'Upload', icon: Camera, route: '/TakePictureScreen' },
    { name: 'Settings', icon: Settings, route: '/SettingsScreen' },
  ];

  return (
    <View style={styles.tabBar}>
      {tabs.map(tab => {
        const Icon = tab.icon;
        const isActive =
          (tab.route === '/(tabs)' && pathname === '/(tabs)') ||
          (tab.route !== '/(tabs)' && pathname === tab.route);
        return (
          <TouchableOpacity
            key={tab.name}
            style={styles.tabBtn}
            onPress={() => router.replace(tab.route as any)}
            accessibilityRole="button"
            accessibilityLabel={tab.name}
          >
            <Icon size={28} color={isActive ? '#f59e0b' : '#9ca3af'} />
            <Text style={[styles.tabLabel, isActive && { color: '#f59e0b' }]}>{tab.name}</Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  tabBar: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
    paddingVertical: 8,
    paddingBottom: 16,
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 100,
  },
  tabBtn: { alignItems: 'center', flex: 1 },
  tabLabel: { fontSize: 12, color: '#9ca3af', marginTop: 2 },
});

export default CustomTabBar; 