import React from 'react';
import { View, Text, StyleSheet, SafeAreaView } from 'react-native';
import CustomTabBar from '../../components/CustomTabBar';

const SettingsScreen = () => (
  <SafeAreaView style={{ flex: 1, backgroundColor: '#fff' }}>
    <View style={[styles.container, { paddingBottom: 96 }]}>
      <Text style={styles.title}>Settings</Text>
      <Text style={styles.placeholder}>Settings options will appear here.</Text>
    </View>
    <CustomTabBar />
  </SafeAreaView>
);

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  title: { fontSize: 24, fontWeight: 'bold', color: '#222', marginBottom: 16 },
  placeholder: { color: '#9ca3af', fontSize: 16 },
});

export default SettingsScreen; 