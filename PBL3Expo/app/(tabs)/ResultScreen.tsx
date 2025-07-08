import React from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity, ScrollView } from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';

const ResultScreen = () => {
  const router = useRouter();
  const params = useLocalSearchParams();
  const result = params.result ? JSON.parse(params.result as string) : null;
  const image = params.image as string | undefined;

  const handleBack = () => {
    router.replace('/(tabs)');
  };

  if (!result) {
    return (
      <View style={styles.center}>
        <Text>No result data.</Text>
        <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
          <Text style={styles.backBtnText}>Back to Home</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Cat Re-Identification Result</Text>
      {image && <Image source={{ uri: image }} style={styles.image} />}
      <View style={styles.resultBox}>
        <Text style={styles.resultTitle}>
          {result.match_found ? '✅ Match Found' : '❌ No Match'}
        </Text>
        <Text style={styles.score}>Score: {result.score ? result.score.toFixed(4) : 'N/A'}</Text>
        {result.matched_id && <Text style={styles.catId}>Cat ID: {result.matched_id}</Text>}
        {result.medical_info && (
          <View style={styles.infoBox}>
            <Text style={styles.infoTitle}>Medical Info:</Text>
            <Text>Name: {result.medical_info.name}</Text>
            <Text>Gender: {result.medical_info.gender}</Text>
            <Text>Vaccinated: {result.medical_info.vaccinated ? 'Yes' : 'No'}</Text>
            <Text>Last Visit: {result.medical_info.last_visit}</Text>
          </View>
        )}
        {result.error && <Text style={styles.error}>Error: {result.error}</Text>}
      </View>
      <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
        <Text style={styles.backBtnText}>Back to Home</Text>
      </TouchableOpacity>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: { flexGrow: 1, alignItems: 'center', justifyContent: 'center', padding: 24, backgroundColor: '#fff' },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#fff' },
  title: { fontSize: 22, fontWeight: 'bold', color: '#222', marginBottom: 24 },
  image: { width: 220, height: 220, borderRadius: 24, marginBottom: 24, backgroundColor: '#f3f4f6' },
  resultBox: { backgroundColor: '#f3f4f6', borderRadius: 16, padding: 20, alignItems: 'center', marginBottom: 24, width: '100%' },
  resultTitle: { fontSize: 20, fontWeight: 'bold', marginBottom: 8 },
  score: { fontSize: 16, marginBottom: 4 },
  catId: { fontSize: 16, marginBottom: 8 },
  infoBox: { marginTop: 12, backgroundColor: '#fff', borderRadius: 12, padding: 12, width: '100%' },
  infoTitle: { fontWeight: 'bold', marginBottom: 4 },
  error: { color: '#ef4444', marginTop: 8 },
  backBtn: { backgroundColor: '#facc15', borderRadius: 16, paddingVertical: 14, paddingHorizontal: 32, alignItems: 'center', marginTop: 24 },
  backBtnText: { color: '#222', fontWeight: 'bold', fontSize: 16 },
});

export default ResultScreen; 