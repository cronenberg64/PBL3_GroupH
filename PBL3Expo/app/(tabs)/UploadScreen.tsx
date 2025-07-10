import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, SafeAreaView, ActivityIndicator, Alert } from 'react-native';
import { Camera } from 'lucide-react-native';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import CustomTabBar from '../../components/CustomTabBar';
import { API_CONFIG, getIdentifyUrl } from '../../config/api';


const UploadScreen = () => {
  const router = useRouter();
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Placeholder for image picker/camera logic
  const handlePickImage = async () => {
    // Ask for permission
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      alert('Camera permission is required!');
      return;
    }
    // Show options: Take photo or pick from gallery
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });
    if (!result.canceled && result.assets && result.assets.length > 0) {
      setImage(result.assets[0].uri);
    }
  };

  const handleUpload = async () => {
    if (!image) return;
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', {
        uri: image,
        type: 'image/jpeg',
        name: 'cat.jpg',
      } as any);

      const response = await fetch(getIdentifyUrl(), {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const result = await response.json();
      setLoading(false);
      router.push({ pathname: '/(tabs)/ResultScreen', params: { result: JSON.stringify(result), image } });
    } catch (e) {
      setLoading(false);
      Alert.alert('Error', 'Failed to connect to server.');
    }
  };

  const handleCancel = () => {
    setImage(null);
    router.replace('/(tabs)'); // Go back to main home page
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#fff' }}>
      <View style={[styles.container, { paddingBottom: 96 }]}>
        <Text style={styles.title}>Upload Cat Photo</Text>
        <TouchableOpacity style={styles.imageBox} onPress={handlePickImage}>
          {image ? (
            <Image source={{ uri: image }} style={styles.image} />
          ) : (
            <Camera color="#facc15" size={48} />
          )}
        </TouchableOpacity>
        <TouchableOpacity style={styles.uploadBtn} onPress={handlePickImage}>
          <Text style={styles.uploadBtnText}>{image ? 'Change Photo' : 'Select Photo'}</Text>
        </TouchableOpacity>
        <View style={styles.buttonGroup}>
          <TouchableOpacity
            style={styles.confirmBtn}
            onPress={handleUpload}
            disabled={!image || loading}
          >
            <Text style={styles.confirmBtnText}>Confirm</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.cancelBtn} onPress={handleCancel} disabled={loading}>
            <Text style={styles.cancelBtnText}>Cancel</Text>
          </TouchableOpacity>
        </View>
        {/* Loading indicator removed; handled by LoadingScreen */}
      </View>
      <CustomTabBar />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  title: { fontSize: 22, fontWeight: 'bold', color: '#222', marginBottom: 32 },
  imageBox: { width: 180, height: 180, borderRadius: 24, backgroundColor: '#f3f4f6', alignItems: 'center', justifyContent: 'center', marginBottom: 24, overflow: 'hidden' },
  image: { width: 180, height: 180, borderRadius: 24 },
  uploadBtn: { backgroundColor: '#facc15', borderRadius: 16, paddingVertical: 14, paddingHorizontal: 32, alignItems: 'center', marginBottom: 16 },
  uploadBtnText: { color: '#222', fontWeight: 'bold', fontSize: 16 },
  buttonGroup: { width: '100%', marginTop: 32, alignItems: 'center' },
  confirmBtn: { backgroundColor: '#fde68a', borderRadius: 16, paddingVertical: 14, alignItems: 'center', marginBottom: 12, width: '70%' },
  confirmBtnText: { color: '#b45309', fontWeight: 'bold', fontSize: 16 },
  cancelBtn: { backgroundColor: '#fde68a', borderRadius: 16, paddingVertical: 14, alignItems: 'center', width: '70%' },
  cancelBtnText: { color: '#b45309', fontWeight: 'bold', fontSize: 16 },
});

export default UploadScreen; 