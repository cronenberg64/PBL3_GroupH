import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, SafeAreaView } from 'react-native';
import { Lock, User } from 'lucide-react-native';
import { useRouter } from 'expo-router';

const LoginScreen = () => {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleLogin = () => {
    // TODO: Integrate with backend authentication
    // For now, just navigate to main app
    router.replace('/(tabs)');
  };

  const handleRegister = () => {
    router.push('/RegisterScreen');
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#fff' }}>
      <View style={styles.container}>
        <Text style={styles.title}>Welcome Back</Text>
        
        {/* Demo Account Info */}
        <Text style={styles.demoText}>Demo: demo@catcare.com / demo123</Text>

        <View style={styles.inputBox}>
          <User color="#888" size={20} style={{ marginRight: 8 }} />
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
        <TouchableOpacity style={styles.loginBtn} onPress={handleLogin}>
          <Text style={styles.loginBtnText}>Login</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={handleRegister} style={{ marginTop: 16 }}>
          <Text style={{ color: '#2563eb' }}>Don't have an account? Register</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  title: { fontSize: 24, fontWeight: 'bold', color: '#222', marginBottom: 24 },
  demoText: { color: '#9ca3af', fontSize: 12, marginBottom: 24 },
  inputBox: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#f3f4f6', borderRadius: 12, paddingHorizontal: 16, marginBottom: 16, width: '100%', maxWidth: 340 },
  input: { flex: 1, height: 48, fontSize: 16 },
  loginBtn: { backgroundColor: '#2563eb', borderRadius: 16, paddingVertical: 14, paddingHorizontal: 32, alignItems: 'center', width: '100%', maxWidth: 340, marginTop: 8 },
  loginBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 16 },
});

export default LoginScreen; 