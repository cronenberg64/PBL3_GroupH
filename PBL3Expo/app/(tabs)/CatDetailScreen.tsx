import React from 'react';
import { View, Text, Image, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { ArrowLeft } from 'lucide-react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';

const CatDetailScreen = () => {
  const router = useRouter();
  const params = useLocalSearchParams();
  
  const cat = params.cat ? JSON.parse(params.cat as string) : {
    id: 'C001',
    name: 'Whiskers',
    breed: 'Persian',
    age: '2 years',
    gender: 'Male',
    image: 'https://images.unsplash.com/photo-1574144611937-0df059b5ef3e?w=300&h=300&fit=crop',
    sterilization: 'Sterilized',
    vaccination: 'Vaccinated',
    medicalHistory: 'Healthy',
  };

  const handleBack = () => {
    router.back();
  };

  const handleMedicalHistory = () => {
    router.push({
      pathname: '/MedicalHistoryScreen',
      params: { cat: JSON.stringify(cat) }
    });
  };

  return (
    <ScrollView style={{ backgroundColor: '#f9fafb' }}>
      <View style={{ position: 'relative' }}>
        <TouchableOpacity style={styles.backButton} onPress={handleBack}>
          <ArrowLeft color="#222" size={20} />
        </TouchableOpacity>
        <Image source={{ uri: cat.image }} style={styles.detailImage} />
      </View>
      <View style={styles.detailCard}>
        <Text style={styles.detailName}>{cat.name}</Text>
        <Text style={styles.detailBreed}>{cat.breed}</Text>
        <View style={styles.detailGridRow}>
          <View style={styles.detailGridItem}><Text style={styles.detailGridLabel}>Cat ID</Text><Text style={styles.detailGridValue}>{cat.id}</Text></View>
          <View style={styles.detailGridItem}><Text style={styles.detailGridLabel}>Gender</Text><Text style={styles.detailGridValue}>{cat.gender}</Text></View>
          <View style={styles.detailGridItem}><Text style={styles.detailGridLabel}>Age</Text><Text style={styles.detailGridValue}>{cat.age}</Text></View>
          <View style={styles.detailGridItem}><Text style={styles.detailGridLabel}>Breed</Text><Text style={styles.detailGridValue}>{cat.breed}</Text></View>
        </View>
        <View style={{ marginBottom: 24 }}>
          <Text style={styles.detailRowText}>Sterilization: {cat.sterilization}</Text>
          <Text style={styles.detailRowText}>Vaccination: {cat.vaccination}</Text>
        </View>
        <TouchableOpacity style={styles.medicalBtn} onPress={handleMedicalHistory}>
          <Text style={styles.medicalBtnText}>Medical History</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  backButton: { position: 'absolute', top: 24, left: 16, zIndex: 10, backgroundColor: '#fff', borderRadius: 20, padding: 8, elevation: 2 },
  detailImage: { width: '100%', height: 240, resizeMode: 'cover' },
  detailCard: { backgroundColor: '#fff', borderTopLeftRadius: 32, borderTopRightRadius: 32, marginTop: -32, padding: 24 },
  detailName: { fontSize: 22, fontWeight: 'bold', color: '#222' },
  detailBreed: { color: '#666', marginBottom: 8 },
  detailGridRow: { flexDirection: 'row', flexWrap: 'wrap', marginBottom: 16 },
  detailGridItem: { width: '48%', backgroundColor: '#f3f4f6', borderRadius: 12, padding: 12, margin: '1%' },
  detailGridLabel: { color: '#888', fontSize: 12 },
  detailGridValue: { fontWeight: 'bold', color: '#222', fontSize: 14 },
  detailRowText: { color: '#222', marginBottom: 8 },
  medicalBtn: { backgroundColor: '#222', borderRadius: 12, paddingVertical: 14, alignItems: 'center', marginBottom: 16 },
  medicalBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 16 },
});

export default CatDetailScreen; 