import React from 'react';
import { View, Text, StyleSheet, SafeAreaView, TouchableOpacity } from 'react-native';
import { Camera, Clock, HelpCircle } from 'lucide-react-native';
import CustomTabBar from '../../components/CustomTabBar';
import { useRouter } from 'expo-router';

const HomeScreen = () => {
  const router = useRouter();
  const handleStartIdentify = () => {
    // To be implemented: navigate to identify page
  };
  const handleScanHistory = () => {
    router.push('/(tabs)/ScanHistoryScreen');
  };
  const handleReport = () => {
    router.push('/(tabs)/ReportScreen');
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#fff' }}>
      <View style={styles.container}>
        {/* Logo */}
        <View style={styles.logoContainer}>
          <Text style={styles.logoText}>
            <Text style={{ color: '#22c55e' }}>か</Text>
            <Text style={{ color: '#f43f5e' }}>る</Text>
            <Text style={{ color: '#f59e42' }}>か</Text>
            <Text style={{ color: '#f43f5e' }}>る</Text>
            <Text> </Text>
            <Text style={{ color: '#facc15', fontSize: 32 }}>🐾</Text>
          </Text>
        </View>

        {/* Account Card */}
        <View style={styles.accountCard}>
          <Text style={styles.accountTitle}>Account</Text>
          <Text style={styles.accountDesc}>Description</Text>
        </View>

        {/* Start Identify Button */}
        <TouchableOpacity style={styles.actionBtn} onPress={handleStartIdentify}>
          <Camera color="#d97706" size={24} style={{ marginRight: 12 }} />
          <Text style={styles.actionBtnText}>Start Identify</Text>
        </TouchableOpacity>

        {/* Scan History Button */}
        <TouchableOpacity style={[styles.actionBtn, styles.scanHistoryBtn]} onPress={handleScanHistory}>
          <Clock color="#d97706" size={24} style={{ marginRight: 12 }} />
          <Text style={styles.actionBtnText}>Scan History</Text>
        </TouchableOpacity>
      </View>
      {/* Report Button */}
      <TouchableOpacity style={styles.reportBtn} onPress={handleReport}>
        <HelpCircle color="#d97706" size={34} />
      </TouchableOpacity>
      <CustomTabBar />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    padding: 24,
    paddingBottom: 96,
    backgroundColor: '#fff',
  },
  logoContainer: {
    alignItems: 'center',
    marginTop: 16,
    marginBottom: 32,
  },
  logoText: {
    fontSize: 32,
    fontWeight: 'bold',
    letterSpacing: 2,
  },
  accountCard: {
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 16,
    padding: 24,
    marginBottom: 40,
    backgroundColor: '#fff',
    alignItems: 'flex-start',
    width: '100%',
    maxWidth: 400,
    minHeight: 180,
    height: 200,
    shadowColor: '#000',
    shadowOpacity: 0.04,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
  },
  accountTitle: { fontWeight: 'bold', fontSize: 18, color: '#444', marginBottom: 4 },
  accountDesc: { color: '#bdbdbd', fontSize: 15 },
  actionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fef3c7',
    borderRadius: 16,
    paddingVertical: 28,
    paddingHorizontal: 24,
    justifyContent: 'center',
    marginBottom: 24,
    width: '100%',
    maxWidth: 400,
    shadowColor: '#facc15',
    shadowOpacity: 0.08,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
  },
  scanHistoryBtn: {
    backgroundColor: '#edfff4',
    shadowColor: '#22c55e',
  },
  actionBtnText: {
    color: '#d97706',
    fontWeight: 'bold',
    fontSize: 20,
  },
  reportBtn: {
    position: 'absolute',
    bottom: 96,
    right: 32,
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#fff',
    borderWidth: 0,
    borderColor: 'transparent',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#facc15',
    shadowOpacity: 0.12,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 2 },
    zIndex: 200,
  },
});

export default HomeScreen; 