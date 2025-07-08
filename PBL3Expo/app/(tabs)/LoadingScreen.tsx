import React, { useEffect } from 'react';
import { View, Text, StyleSheet, Image, Alert } from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { API_CONFIG } from '../../config/api';

const LoadingScreen = () => {
  const router = useRouter();
  const params = useLocalSearchParams();
  const image = params.image as string | undefined;

  useEffect(() => {
    const identifyCat = async () => {
      if (!image) return;
      try {
        const formData = new FormData();
        formData.append('image', {
          uri: image,
          type: 'image/jpeg',
          name: 'cat.jpg',
        } as any);
        const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.IDENTIFY}`, {
          method: 'POST',
          headers: { 'Content-Type': 'multipart/form-data' },
          body: formData,
        });
        const result = await response.json();
        if (response.ok) {
          router.replace({ pathname: '/(tabs)/ResultScreen', params: { result: JSON.stringify(result), image } });
        } else {
          Alert.alert('Error', result.error || 'Failed to identify cat.');
          router.replace('/(tabs)/UploadScreen');
        }
      } catch (e) {
        Alert.alert('Error', 'Failed to connect to server.');
        router.replace('/(tabs)/UploadScreen');
      }
    };
    identifyCat();
  }, [image]);

  return (
    <View style={styles.container}>
      <Text style={styles.waitText}>Please wait...</Text>
      <Image source={require('../../assets/catWalking.gif')} style={styles.catGif} resizeMode="contain" />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  waitText: {
    color: '#f59e0b',
    fontSize: 18,
    marginBottom: 32,
    marginTop: 32,
    fontWeight: '500',
  },
  catGif: {
    width: 180,
    height: 180,
  },
});

export default LoadingScreen; 