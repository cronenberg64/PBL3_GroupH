import React from 'react';
import { View, Text, StyleSheet, Image, TouchableOpacity, SafeAreaView } from 'react-native';
import { LogOut } from 'lucide-react-native';
import { useRouter } from 'expo-router';

const user = {
  name: 'Jonathan Setiawan',
  email: 'jonathan@example.com',
  avatar: 'https://randomuser.me/api/portraits/men/32.jpg',
};

const ProfileScreen = () => {
  const router = useRouter();

  const handleLogout = () => {
    // TODO: Clear authentication state
    router.replace('/LoginScreen');
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#fff' }}>
      <View style={styles.container}>
        <Image source={{ uri: user.avatar }} style={styles.avatar} />
        <Text style={styles.name}>{user.name}</Text>
        <Text style={styles.email}>{user.email}</Text>
        <TouchableOpacity style={styles.logoutBtn} onPress={handleLogout}>
          <LogOut color="#fff" size={20} />
          <Text style={styles.logoutText}>Logout</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  avatar: { width: 100, height: 100, borderRadius: 50, marginBottom: 24 },
  name: { fontSize: 22, fontWeight: 'bold', color: '#222', marginBottom: 4 },
  email: { color: '#666', marginBottom: 32 },
  logoutBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#ef4444', borderRadius: 16, paddingVertical: 14, paddingHorizontal: 32, marginTop: 24 },
  logoutText: { color: '#fff', fontWeight: 'bold', fontSize: 16, marginLeft: 8 },
});

export default ProfileScreen; 