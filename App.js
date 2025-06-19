import React, { useState } from 'react';
import { View, Text, Button, Image, StyleSheet, ActivityIndicator, ScrollView } from 'react-native';
import axios from 'axios';
import { launchCamera } from 'react-native-image-picker';

const FLASK_API_URL = "http://172.31.202.213:5000/identify"; // ← あなたのFlaskサーバーURL

export default function App() {
  const [photo, setPhoto] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleCamera = () => {
    launchCamera({ mediaType: 'photo', cameraType: 'back' }, (response) => {
      if (response.didCancel || response.errorCode) {
        setError('Camera cancelled or failed');
        return;
      }

      const asset = response.assets[0];
      setPhoto(asset);
      uploadImage(asset);
    });
  };

  const uploadImage = async (image) => {
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('image', {
      uri: image.uri,
      type: image.type,
      name: image.fileName || 'photo.jpg',
    });

    try {
      const response = await axios.post(FLASK_API_URL, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (err) {
      setError('Failed to connect to server.');
    } finally {
      setLoading(false);
    }
  };

  const renderResult = () => {
    if (!result) return null;

    return (
      <View style={styles.resultBox}>
        <Text style={styles.resultTitle}>
          {result.match_found ? '✅ 既知猫（登録済み）' : '❌ 未登録の猫'}
        </Text>
        <Text>信頼度: {(result.score * 100).toFixed(2)}%</Text>

        {result.medical_info && (
          <View style={styles.infoBox}>
            <Text>📋 医療情報:</Text>
            <Text>名前: {result.medical_info.name}</Text>
            <Text>性別: {result.medical_info.gender}</Text>
            <Text>ワクチン: {result.medical_info.vaccinated ? '済み' : '未接種'}</Text>
            <Text>最終受診日: {result.medical_info.last_visit}</Text>
          </View>
        )}
      </View>
    );
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>🐱 猫の再識別アプリ</Text>

      <Button title="📷 猫を撮影" onPress={handleCamera} />

      {photo && (
        <Image source={{ uri: photo.uri }} style={styles.image} resizeMode="contain" />
      )}

      {loading && <ActivityIndicator size="large" color="#007AFF" style={{ marginTop: 20 }} />}

      {error && <Text style={styles.error}>{error}</Text>}

      {renderResult()}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flexGrow: 1, alignItems: 'center', padding: 20 },
  title: { fontSize: 24, marginBottom: 20 },
  image: { width: 300, height: 300, marginTop: 20, borderRadius: 10 },
  resultBox: { marginTop: 30, padding: 15, backgroundColor: '#e0f7fa', borderRadius: 10 },
  resultTitle: { fontSize: 20, marginBottom: 10 },
  infoBox: { marginTop: 10 },
  error: { color: 'red', marginTop: 20 },
});
