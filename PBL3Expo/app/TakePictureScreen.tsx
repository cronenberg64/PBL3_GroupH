import React, { useRef, useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, SafeAreaView, Platform } from 'react-native';
import { useRouter } from 'expo-router';
import { useCameraPermissions } from 'expo-camera';

let Camera: any, useCameraPermissionsNative: (() => [any, any]) | undefined;
if (Platform.OS !== 'web') {
  const expoCamera = require('expo-camera');
  Camera = expoCamera.Camera;
  useCameraPermissionsNative = expoCamera.useCameraPermissions;
}

const TakePictureScreen = () => {
  const [permission, requestPermission] = Platform.OS !== 'web' && useCameraPermissionsNative ? useCameraPermissionsNative() : [null, null];
  const [photo, setPhoto] = useState<string | null>(null);
  const cameraRef = useRef<any>(null);
  const router = useRouter();

  React.useEffect(() => {
    if (Platform.OS === 'web' || !permission) return;
    if (!permission.granted) {
      requestPermission();
    }
  }, [permission]);

  const takePicture = async () => {
    if (cameraRef.current && cameraRef.current.takePictureAsync) {
      const result = await cameraRef.current.takePictureAsync();
      setPhoto(result.uri);
    }
  };

  const handleConfirm = () => {
    router.replace('/(tabs)');
  };

  const handleCancel = () => {
    setPhoto(null);
  };

  if (Platform.OS === 'web') {
    return <View style={styles.center}><Text>Camera is only available on mobile devices.</Text></View>;
  }

  if (!permission) {
    return <View style={styles.center}><Text>Requesting camera permission...</Text></View>;
  }
  if (!permission.granted) {
    return <View style={styles.center}><Text>No access to camera</Text></View>;
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#000' }}>
      <View style={styles.cameraContainer}>
        {photo ? (
          <View style={styles.previewContainer}>
            <Image source={{ uri: photo }} style={styles.preview} />
            <View style={styles.previewActions}>
              <TouchableOpacity style={styles.cancelBtn} onPress={handleCancel}>
                <Text style={styles.cancelText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.confirmBtn} onPress={handleConfirm}>
                <Text style={styles.confirmText}>Confirm</Text>
              </TouchableOpacity>
            </View>
          </View>
        ) : (
          Camera && (
            <Camera style={styles.camera} type={0} ref={cameraRef}>
              <View style={styles.captureContainer}>
                <TouchableOpacity style={styles.captureBtn} onPress={takePicture} />
              </View>
            </Camera>
          )
        )}
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  cameraContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  camera: { flex: 1, width: '100%', justifyContent: 'flex-end' },
  captureContainer: { flex: 1, justifyContent: 'flex-end', alignItems: 'center', marginBottom: 36 },
  captureBtn: { width: 72, height: 72, borderRadius: 36, backgroundColor: '#fff', borderWidth: 4, borderColor: '#facc15' },
  previewContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', width: '100%' },
  preview: { width: '100%', height: '80%', borderRadius: 16, marginBottom: 24 },
  previewActions: { flexDirection: 'row', justifyContent: 'space-around', width: '100%' },
  cancelBtn: { backgroundColor: '#fff', padding: 16, borderRadius: 12, marginHorizontal: 16 },
  confirmBtn: { backgroundColor: '#facc15', padding: 16, borderRadius: 12, marginHorizontal: 16 },
  cancelText: { color: '#222', fontWeight: 'bold' },
  confirmText: { color: '#fff', fontWeight: 'bold' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#fff' },
});

export default TakePictureScreen; 