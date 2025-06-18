import React, { useState } from 'react';
import { View, Button, Image, Text, StyleSheet, ScrollView } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';

export default function App() {
  const [photo, setPhoto] = useState(null);
  const [result, setResult] = useState(null);

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      alert('Camera access is required.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setPhoto(result.assets[0]);
      setResult(null);
    }
  };

  const identifyCat = async () => {
    if (!photo) return;

    const uriParts = photo.uri.split('.');
    const fileType = uriParts[uriParts.length - 1];

    const formData = new FormData();
    formData.append('image', {
      uri: photo.uri,
      name: `cat.${fileType}`,
      type: `image/${fileType}`,
    });

    try {
      const res = await axios.post('http://192.168.1.10:5000/identify', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (res.data.match_found) {
        setResult({
          matched_id: `既知猫: ${res.data.matched_id}`,
          score: `信頼度: ${res.data.score.toFixed(2)}`,
          medical_info: res.data.medical_info,
        });
      } else {
        setResult({
          matched_id: '未登録猫です。',
          score: `照合スコア: ${res.data.score.toFixed(2)}`,
          medical_info: null,
        });
      }
    } catch (err) {
      setResult({ error: err.message });
    }
  };

  return (
    <View style={styles.container}>
      <Button title="猫の写真を撮影" onPress={takePhoto} />

      {photo && (
        <>
          <Image source={{ uri: photo.uri }} style={styles.image} />
          <Button title="照合する" onPress={identifyCat} />
        </>
      )}

      {result && (
        <ScrollView style={styles.result}>
          <Text style={styles.resultText}>結果:</Text>
          <Text style={styles.resultText}>{result.matched_id}</Text>
          <Text style={styles.resultText}>{result.score}</Text>
          {result.medical_info && (
            <View style={styles.medicalInfo}>
              <Text style={styles.resultText}>医療情報:</Text>
              <Text style={styles.resultText}>登録番号: {result.medical_info.registration_number}</Text>
              <Text style={styles.resultText}>ワクチン接種履歴: {result.medical_info.vaccine_history}</Text>
              <Text style={styles.resultText}>性別: {result.medical_info.gender}</Text>
            </View>
          )}
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 20 },
  image: { width: 300, height: 300, marginTop: 20, resizeMode: 'contain' },
  result: { marginTop: 20, width: '100%' },
  resultText: { fontSize: 16, marginBottom: 10, textAlign: 'center' },
  medicalInfo: { marginTop: 20, padding: 10, backgroundColor: '#f0f0f0' },
});
