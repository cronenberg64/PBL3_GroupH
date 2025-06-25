import React from 'react';
import { View, Text, StyleSheet, SafeAreaView, TouchableOpacity, Image, ScrollView } from 'react-native';
import { Camera } from 'lucide-react-native';
import CustomTabBar from '../../components/CustomTabBar';

const styles = StyleSheet.create({
  scroll: { padding: 24, paddingBottom: 48 },
  logoContainer: { alignItems: 'center', marginBottom: 24 },
  logo: { fontSize: 32, fontWeight: 'bold', letterSpacing: 2 },
  accountCard: {
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 12,
    padding: 24,
    marginBottom: 32,
    backgroundColor: '#fff',
    alignItems: 'flex-start',
  },
  accountTitle: { fontWeight: 'bold', fontSize: 18, color: '#222', marginBottom: 4 },
  accountDesc: { color: '#9ca3af', fontSize: 15 },
  identifyBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fde68a',
    borderRadius: 16,
    paddingVertical: 24,
    justifyContent: 'center',
    marginBottom: 32,
  },
  identifyBtnText: { color: '#f59e0b', fontWeight: 'bold', fontSize: 20 },
  historyCard: {
    borderWidth: 1,
    borderColor: '#d1d5db',
    borderRadius: 8,
    backgroundColor: '#fff',
    padding: 16,
    marginBottom: 24,
  },
  historyTitle: { fontWeight: 'bold', fontSize: 18, color: '#222', marginBottom: 8 },
  historyRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 16 },
  historyImageBox: {
    width: 48,
    height: 48,
    borderRadius: 8,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  historyImage: { width: 32, height: 32, tintColor: '#d1d5db' },
  historyFrame: { color: '#9ca3af', fontWeight: 'bold', fontSize: 14 },
  historyMeta: { color: '#222', fontSize: 13, marginTop: 2 },
  historyPercent: { color: '#222', fontSize: 18, fontWeight: 'bold', marginTop: 2 },
});

const logoText = (
  <Text style={styles.logo}>
    <Text style={{ color: '#22c55e' }}>か</Text>
    <Text style={{ color: '#f43f5e' }}>る</Text>
    <Text style={{ color: '#f59e42' }}>か</Text>
    <Text style={{ color: '#f43f5e' }}>る</Text>
    <Text style={{ color: '#facc15' }}>🐾</Text>
  </Text>
);

const history = [
  {
    date: '20**/**/**',
    location: 'Osaka, Ibaraki',
    percent: null,
    image: null,
  },
  {
    date: '20**/**/**',
    location: null,
    percent: null,
    image: null,
  },
];

const HomeScreen = () => {
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#fff' }}>
      <ScrollView contentContainerStyle={[styles.scroll, { paddingBottom: 96 }]}>
        {/* Logo */}
        <View style={styles.logoContainer}>{logoText}</View>

        {/* Account Card */}
        <View style={styles.accountCard}>
          <Text style={styles.accountTitle}>Account</Text>
          <Text style={styles.accountDesc}>Description</Text>
        </View>

        {/* Start Identify Button */}
        <TouchableOpacity style={styles.identifyBtn}>
          <Camera color="#f59e0b" size={24} style={{ marginRight: 12 }} />
          <Text style={styles.identifyBtnText}>Start Identify</Text>
        </TouchableOpacity>

        {/* History Section */}
        <View style={styles.historyCard}>
          <Text style={styles.historyTitle}>History</Text>
          {history.map((item, idx) => (
            <View key={idx} style={styles.historyRow}>
              <View style={styles.historyImageBox}>
                <Camera color="#d1d5db" size={32} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.historyFrame}>Frame</Text>
                <Text style={styles.historyMeta}>
                  Searched: {item.date}
                  {item.location ? `   Location: ${item.location}` : ''}
                </Text>
                {item.percent !== null ? (
                  <Text style={styles.historyPercent}>{item.percent}%</Text>
                ) : (
                  <Text style={styles.historyPercent}>___ %</Text>
                )}
              </View>
            </View>
          ))}
        </View>
      </ScrollView>
      <CustomTabBar />
    </SafeAreaView>
  );
};

export default HomeScreen; 