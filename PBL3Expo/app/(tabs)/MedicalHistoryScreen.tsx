import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { ArrowLeft } from 'lucide-react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';

const medicalHistory = [
  { date: '2024-03-15', hospital: 'Animal Hospital', treatment: 'Vaccination', notes: 'Regular checkup, healthy.' },
  { date: '2023-09-10', hospital: 'Animal Hospital', treatment: 'Sterilization', notes: 'Surgery and after-care.' },
];

const MedicalHistoryScreen = () => {
  const router = useRouter();
  const params = useLocalSearchParams();
  
  const cat = params.cat ? JSON.parse(params.cat as string) : {
    name: 'Whiskers'
  };

  const handleBack = () => {
    router.back();
  };

  return (
    <ScrollView style={{ backgroundColor: '#f9fafb' }}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={handleBack}>
          <ArrowLeft color="#222" size={20} />
        </TouchableOpacity>
        <Text style={styles.title}>Medical History - {cat.name}</Text>
      </View>
      <View style={styles.list}>
        {medicalHistory.map((item, idx) => (
          <View key={idx} style={styles.item}>
            <Text style={styles.itemDate}>{item.date}</Text>
            <Text style={styles.itemHospital}>{item.hospital}</Text>
            <Text style={styles.itemTreatment}>{item.treatment}</Text>
            <Text style={styles.itemNotes}>{item.notes}</Text>
          </View>
        ))}
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  header: { flexDirection: 'row', alignItems: 'center', padding: 16, backgroundColor: '#fff', borderBottomWidth: 1, borderBottomColor: '#eee' },
  backButton: { marginRight: 12, backgroundColor: '#f3f4f6', borderRadius: 20, padding: 8 },
  title: { fontSize: 20, fontWeight: 'bold', color: '#222' },
  list: { padding: 16 },
  item: { backgroundColor: '#fff', borderRadius: 16, padding: 16, marginBottom: 12, elevation: 1 },
  itemDate: { fontWeight: 'bold', color: '#222', marginBottom: 4 },
  itemHospital: { color: '#666', marginBottom: 2 },
  itemTreatment: { color: '#2563eb', marginBottom: 2 },
  itemNotes: { color: '#888' },
});

export default MedicalHistoryScreen; 