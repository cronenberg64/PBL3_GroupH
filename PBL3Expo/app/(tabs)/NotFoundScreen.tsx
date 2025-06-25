import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, SafeAreaView } from 'react-native';
import { AlertTriangle } from 'lucide-react-native';
import { useRouter } from 'expo-router';

const NotFoundScreen = () => {
  const router = useRouter();

  const handleGoHome = () => {
    router.replace('/(tabs)');
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#fff' }}>
      <View style={styles.container}>
        <AlertTriangle color="#f87171" size={48} />
        <Text style={styles.title}>Page Not Found</Text>
        <Text style={styles.desc}>Sorry, the page you are looking for does not exist.</Text>
        <TouchableOpacity style={styles.homeBtn} onPress={handleGoHome}>
          <Text style={styles.homeBtnText}>Go to Home</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  title: { fontSize: 24, fontWeight: 'bold', color: '#222', marginTop: 24, marginBottom: 8 },
  desc: { color: '#666', marginBottom: 32, textAlign: 'center' },
  homeBtn: { backgroundColor: '#2563eb', borderRadius: 16, paddingVertical: 14, paddingHorizontal: 32, alignItems: 'center' },
  homeBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 16 },
});

export default NotFoundScreen; 