import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, Image, Alert } from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { API_CONFIG } from '../../config/api';

const LoadingScreen = () => {
  const router = useRouter();
  const params = useLocalSearchParams();
  const image = params.image as string | undefined;
  const [dots, setDots] = useState('');

  // Animate the three dots
  useEffect(() => {
    const interval = setInterval(() => {
      setDots(prev => (prev.length < 3 ? prev + '.' : ''));
    }, 500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const identifyCat = async () => {
      if (!image) return;
      try {
        const formData = new FormData();
        // Try to infer image type from uri
        let fileType = 'image/jpeg';
        if (image.endsWith('.png')) fileType = 'image/png';
        else if (image.endsWith('.jpg') || image.endsWith('.jpeg')) fileType = 'image/jpeg';
        formData.append('image', {
          uri: image,
          type: fileType,
          name: 'cat' + (fileType === 'image/png' ? '.png' : '.jpg'),
        } as any);
        console.log('Uploading image:', { uri: image, type: fileType, name: 'cat' + (fileType === 'image/png' ? '.png' : '.jpg') });
        const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.IDENTIFY}`, {
          method: 'POST',
          // Do NOT set Content-Type header manually for FormData in React Native
          body: formData,
        });
        console.log('Response status:', response.status);
        console.log('Response body:', await response.text());
        const result = await response.json();
        if (response.ok) {
          router.replace({ pathname: '/(tabs)/ResultScreen', params: { result: JSON.stringify(result), image } });
        } else {
          Alert.alert('Error', result.error || 'Failed to identify cat.');
          router.replace('/(tabs)/UploadScreen');
        }
      } catch (e) {
        console.log('Upload error:', e);
        Alert.alert('Error', 'Failed to connect to server.');
        router.replace('/(tabs)/UploadScreen');
      }
    };
    identifyCat();
  }, [image]);

  useEffect(() => {
    fetch('http://YOUR_NGROK_URL/')
      .then(res => res.text())
      .then(data => console.log('Backend response:', data))
      .catch(err => console.log('Fetch error:', err));
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.waitText}>Please wait{dots}</Text>
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