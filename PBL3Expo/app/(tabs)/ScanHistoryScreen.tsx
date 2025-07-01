import React from 'react';
import { View, Text, StyleSheet, SafeAreaView, TextInput, TouchableOpacity } from 'react-native';
import { Search, X, Calendar, ChevronDown } from 'lucide-react-native';
import { useRouter } from 'expo-router';
import CustomTabBar from '../../components/CustomTabBar';

const ScanHistoryScreen = () => {
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
        <Text style={styles.title}>Scan History</Text>

        {/* Search and Filters Card */}
        <View style={styles.searchCard}>
          <View style={styles.searchRow}>
            <Search color="#bdbdbd" size={20} style={{ marginRight: 8 }} />
            <TextInput
              style={styles.searchInput}
              placeholder="Search"
              placeholderTextColor="#bdbdbd"
            />
            <TouchableOpacity>
              <X color="#bdbdbd" size={20} />
            </TouchableOpacity>
          </View>
          <View style={styles.filterRow}>
            <View style={styles.platformBox}>
              <Text style={styles.platformLabel}>Platform</Text>
              <TouchableOpacity style={styles.dropdownBtn}>
                <Text style={styles.dropdownText}>All</Text>
                <ChevronDown color="#bdbdbd" size={16} />
              </TouchableOpacity>
            </View>
            <TouchableOpacity style={styles.dateBtn}>
              <Calendar color="#bdbdbd" size={18} style={{ marginRight: 6 }} />
              <Text style={styles.dateText}>Today</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* No Search History Card */}
        <View style={styles.noHistoryCard}>
          <Text style={styles.noHistoryTitle}>No Search History</Text>
          <Text style={styles.noHistoryDesc}>Haven't Scanned Yet.{"\n"}Start Your First Scan !</Text>
        </View>
      </View>
      <CustomTabBar />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    padding: 24,
    backgroundColor: '#fff',
  },
  logoContainer: {
    alignItems: 'center',
    marginTop: 8,
    marginBottom: 16,
  },
  logoText: {
    fontSize: 32,
    fontWeight: 'bold',
    letterSpacing: 2,
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#222',
    marginBottom: 24,
  },
  searchCard: {
    width: '100%',
    maxWidth: 420,
    backgroundColor: '#fff',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    padding: 16,
    marginBottom: 32,
    shadowColor: '#000',
    shadowOpacity: 0.03,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 2 },
  },
  searchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    paddingHorizontal: 12,
    marginBottom: 12,
  },
  searchInput: {
    flex: 1,
    height: 40,
    fontSize: 16,
    color: '#222',
  },
  filterRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
  },
  platformBox: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 12,
  },
  platformLabel: {
    color: '#bdbdbd',
    fontSize: 15,
    marginRight: 8,
  },
  dropdownBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    paddingHorizontal: 12,
    height: 36,
    marginRight: 8,
  },
  dropdownText: {
    color: '#222',
    fontSize: 15,
    marginRight: 4,
  },
  dateBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#e5e7eb',
    borderRadius: 8,
    paddingHorizontal: 12,
    height: 36,
  },
  dateText: {
    color: '#222',
    fontSize: 15,
  },
  noHistoryCard: {
    width: '100%',
    maxWidth: 420,
    backgroundColor: '#fff',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    padding: 32,
    alignItems: 'center',
    marginTop: 24,
    shadowColor: '#000',
    shadowOpacity: 0.03,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 2 },
  },
  noHistoryTitle: {
    fontWeight: 'bold',
    fontSize: 18,
    color: '#888',
    marginBottom: 8,
  },
  noHistoryDesc: {
    color: '#bdbdbd',
    fontSize: 16,
    textAlign: 'center',
  },
});

export default ScanHistoryScreen; 