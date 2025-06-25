import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, SafeAreaView } from 'react-native';
import { Camera } from 'lucide-react-native';
import { useRouter } from 'expo-router';
import CustomTabBar from '../../components/CustomTabBar';

const UploadScreen = () => {
  const router = useRouter();
  const [image, setImage] = useState<string | null>(null);

  // Placeholder for image picker/camera logic
  const handlePickImage = () => {
    // TODO: Integrate with expo-image-picker or camera
    setImage('https://images.unsplash.com/photo-1574144611937-0df059b5ef3e?w=300&h=300&fit=crop');
  };

  const handleUpload = () => {
    if (image) {
      // TODO: Upload image to backend
      // For now, just navigate back to home
      router.replace('/(tabs)');
    }
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
        {image && (
          <TouchableOpacity style={styles.submitBtn} onPress={handleUpload}>
            <Text style={styles.submitBtnText}>Upload & Identify</Text>
          </TouchableOpacity>
        )}
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
  submitBtn: { backgroundColor: '#2563eb', borderRadius: 16, paddingVertical: 14, paddingHorizontal: 32, alignItems: 'center' },
  submitBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 16 },
});

export default UploadScreen; 