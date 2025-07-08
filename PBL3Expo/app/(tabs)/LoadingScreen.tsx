import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, Image, Alert } from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { API_CONFIG } from '../../config/api';

const LoadingScreen = () => {
  // This screen is no longer used for backend upload. You may remove or repurpose it.
  return (
    <View style={styles.container}>
      <Text style={styles.waitText}>Processing...</Text>
      <Image source={require('../../assets/catWalking.gif')} style={styles.catGif} resizeMode="contain" />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  waitText: {
    color: '#f59e0b',
    fontSize: 18,
    marginBottom: 32,
    marginTop: 32,
    fontWeight: '500',
  },
  catGif: {
    width: 180,
    height: 180,
  },
});

export default LoadingScreen; 