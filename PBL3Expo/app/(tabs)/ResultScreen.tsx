import React from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity, ScrollView, Alert } from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import CustomTabBar from '../../components/CustomTabBar';

function getConfidenceLabel(score: number) {
  if (score >= 0.8) return 'Very Likely Match';
  if (score >= 0.6) return 'Possible Match – Needs Confirmation';
  return 'No reliable match';
}

function distanceToSimilarity(distance: number) {
  // For deep learning embeddings, distance is cosine distance (0-2)
  // Convert to similarity (0-1, higher is better)
  let sim = 1 - distance;
  if (sim < 0) sim = 0;
  if (sim > 1) sim = 1;
  return sim;
}

function toPercent(score: number) {
  // Clamp and convert to 0-100%
  return Math.round(Math.max(0, Math.min(1, score)) * 100);
}

const ResultScreen = () => {
  const router = useRouter();
  const params = useLocalSearchParams();
  const result = params.result ? JSON.parse(params.result as string) : null;
  const image = params.image as string | undefined;

  const handleBack = () => {
    router.replace('/(tabs)');
  };

  const handleViewProfile = () => {
    // TODO: Implement navigation to profile
  };

  const handleConfirm = () => {
    Alert.alert('Confirmed', 'You have confirmed this match.');
  };
  const handleReject = () => {
    Alert.alert('Rejected', 'You have rejected this match.');
  };
  const handleReport = () => {
    Alert.alert('Reported', 'You have reported this match.');
  };

  if (!result) {
    return (
      <View style={styles.center}>
        <Text>No result data.</Text>
        <TouchableOpacity style={styles.backBtn} onPress={handleBack}>
          <Text style={styles.backBtnText}>Back to Home</Text>
        </TouchableOpacity>
        <CustomTabBar />
      </View>
    );
  }

  // Convert backend distance to similarity (0-1, higher is better)
  let similarity = 0;
  if (result.score !== undefined && result.score !== null) {
    similarity = distanceToSimilarity(result.score);
  }
  const confidence = toPercent(similarity); // This is the % you display
  const label = getConfidenceLabel(similarity);
  const isHigh = confidence >= 80;
  const isModerate = confidence >= 60 && confidence < 80;
  const isLow = confidence < 60;

  return (
    <View style={styles.page}>
      <View style={styles.headerRow}>
        <TouchableOpacity onPress={handleBack} style={styles.backIcon}>
          <Text style={{ fontSize: 24, color: '#bfa14a' }}>{'<'}</Text>
        </TouchableOpacity>
        <Text style={styles.header}>Result</Text>
      </View>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.confidence}>{confidence}%</Text>
        <Text style={styles.confidenceLabel}>{label}</Text>
        <View style={styles.card}>
          {image ? (
            <Image source={{ uri: image }} style={styles.catImage} />
          ) : (
            <View style={[styles.catImage, styles.placeholderImg]} />
          )}
          <Text style={styles.catId}>{result.id ? result.id : 'Cat ID'}</Text>
          <Text style={styles.info}>{result.info ? result.info : 'Information'}</Text>
        </View>
        {isHigh && (
          <TouchableOpacity style={styles.profileBtn} onPress={handleViewProfile}>
            <Text style={styles.profileBtnText}>View Profile</Text>
          </TouchableOpacity>
        )}
        {isModerate && (
          <View style={styles.actionRow}>
            <TouchableOpacity style={styles.actionBtn} onPress={handleConfirm}>
              <Text style={styles.actionBtnText}>Confirm</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.actionBtn} onPress={handleReject}>
              <Text style={styles.actionBtnText}>Reject</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.actionBtn} onPress={handleReport}>
              <Text style={styles.actionBtnText}>Report</Text>
            </TouchableOpacity>
          </View>
        )}
        {isLow && (
          <TouchableOpacity style={styles.actionBtn} onPress={handleReport}>
            <Text style={styles.actionBtnText}>Report False Re-ID</Text>
          </TouchableOpacity>
        )}
      </ScrollView>
      <CustomTabBar />
    </View>
  );
};

const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: '#fafafa' },
  headerRow: { flexDirection: 'row', alignItems: 'center', paddingTop: 32, paddingBottom: 8, backgroundColor: '#fff', borderBottomWidth: 1, borderBottomColor: '#eee' },
  backIcon: { paddingHorizontal: 16 },
  header: { fontSize: 18, color: '#aaa', fontWeight: '600', marginLeft: 0 },
  container: { alignItems: 'center', padding: 24 },
  confidence: { fontSize: 48, color: '#bfa14a', fontWeight: 'bold', marginTop: 24 },
  confidenceLabel: { color: '#bfa14a', fontSize: 16, marginBottom: 24 },
  card: { backgroundColor: '#fff', borderRadius: 16, padding: 20, alignItems: 'center', marginBottom: 24, width: 240, shadowColor: '#000', shadowOpacity: 0.05, shadowRadius: 8, elevation: 2 },
  catImage: { width: 140, height: 140, borderRadius: 8, backgroundColor: '#eee', marginBottom: 12 },
  placeholderImg: { justifyContent: 'center', alignItems: 'center' },
  catId: { fontSize: 16, fontWeight: 'bold', marginBottom: 4 },
  info: { fontSize: 14, color: '#888' },
  profileBtn: { backgroundColor: '#eab308', borderRadius: 8, paddingVertical: 16, alignItems: 'center', width: 220, marginTop: 8 },
  profileBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 16 },
  actionRow: { flexDirection: 'row', justifyContent: 'center', marginTop: 12 },
  actionBtn: { backgroundColor: '#eab308', borderRadius: 8, paddingVertical: 12, paddingHorizontal: 16, alignItems: 'center', marginHorizontal: 6 },
  actionBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 14 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#fff' },
  backBtn: { backgroundColor: '#facc15', borderRadius: 16, paddingVertical: 14, paddingHorizontal: 32, alignItems: 'center', marginTop: 24 },
  backBtnText: { color: '#222', fontWeight: 'bold', fontSize: 16 },
});

export default ResultScreen; 