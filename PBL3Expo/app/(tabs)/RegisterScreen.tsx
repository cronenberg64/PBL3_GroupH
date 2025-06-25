import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, SafeAreaView } from 'react-native';
import { Lock, User, Mail } from 'lucide-react-native';
import { useRouter } from 'expo-router';

const RegisterScreen = () => {
  const router = useRouter();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleRegister = () => {
    // TODO: Integrate with backend registration
    // For now, just navigate to main app
    router.replace('/(tabs)');
  };

  const handleLogin = () => {
    router.push('/LoginScreen');
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#fff' }}>
      <View style={styles.container}>
        <Text style={styles.title}>Create Account</Text>
        <View style={styles.inputBox}>
          <User color="#888" size={20} style={{ marginRight: 8 }} />
          <TextInput
            style={styles.input}
            placeholder="Name"
            value={name}
            onChangeText={setName}
          />
        </View>
        <View style={styles.inputBox}>
          <Mail color="#888" size={20} style={{ marginRight: 8 }} />
          <TextInput
            style={styles.input}
            placeholder="Email"
            value={email}
            onChangeText={setEmail}
            autoCapitalize="none"
            keyboardType="email-address"
          />
        </View>
        <View style={styles.inputBox}>
          <Lock color="#888" size={20} style={{ marginRight: 8 }} />
          <TextInput
            style={styles.input}
            placeholder="Password"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
          />
        </View>
        <TouchableOpacity style={styles.registerBtn} onPress={handleRegister}>
          <Text style={styles.registerBtnText}>Register</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={handleLogin} style={{ marginTop: 16 }}>
          <Text style={{ color: '#2563eb' }}>Already have an account? Login</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  title: { fontSize: 24, fontWeight: 'bold', color: '#222', marginBottom: 32 },
  inputBox: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#f3f4f6', borderRadius: 12, paddingHorizontal: 16, marginBottom: 16, width: '100%', maxWidth: 340 },
  input: { flex: 1, height: 48, fontSize: 16 },
  registerBtn: { backgroundColor: '#facc15', borderRadius: 16, paddingVertical: 14, paddingHorizontal: 32, alignItems: 'center', width: '100%', maxWidth: 340, marginTop: 8 },
  registerBtnText: { color: '#222', fontWeight: 'bold', fontSize: 16 },
});

export default RegisterScreen; 