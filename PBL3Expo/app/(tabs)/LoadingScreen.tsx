import React, { useEffect, useState, useRef } from 'react';
import { View, Text, StyleSheet, Image, Alert, ActivityIndicator, Animated, Easing } from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { API_CONFIG, getIdentifyUrl } from '../../config/api';
import { Loader, CheckCircle, XCircle } from 'lucide-react-native';

const LoadingScreen = () => {
  const router = useRouter();
  const params = useLocalSearchParams();
  const imageUri = params.imageUri as string;
  const action = params.action as string;
  
  const [loadingState, setLoadingState] = useState<'processing' | 'completed' | 'error'>('processing');
  const [progressText, setProgressText] = useState('Analyzing cat image...');
  const [result, setResult] = useState<any>(null);

  // Animation for Loader icon
  const spinValue = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    if (loadingState === 'processing') {
      Animated.loop(
        Animated.timing(spinValue, {
          toValue: 1,
          duration: 1000,
          easing: Easing.linear,
          useNativeDriver: true,
        })
      ).start();
    } else {
      spinValue.stopAnimation();
    }
  }, [loadingState]);
  const spin = spinValue.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  // Reset state on navigation change (new image or action)
  useEffect(() => {
    setLoadingState('processing');
    setProgressText('Analyzing cat image...');
    setResult(null);
  }, [imageUri, action]);

  useEffect(() => {
    if (action === 'identify' && imageUri) {
      performIdentification();
    }
  }, [action, imageUri]);

  const performIdentification = async () => {
    try {
      setProgressText('Uploading image to server...');
      await new Promise(resolve => setTimeout(resolve, 1000));
      setProgressText('Detecting cat in image...');
      await new Promise(resolve => setTimeout(resolve, 1500));
      setProgressText('Extracting features...');
      await new Promise(resolve => setTimeout(resolve, 1000));
      setProgressText('Matching against database...');
      await new Promise(resolve => setTimeout(resolve, 1500));

      const formData = new FormData();
      formData.append('image', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'cat.jpg',
      } as any);

      const response = await fetch(getIdentifyUrl(), {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const apiResult = await response.json();
      setResult(apiResult);
      setProgressText('Analysis complete!');
      setLoadingState('completed');
      setTimeout(() => {
        router.replace({ 
          pathname: '/(tabs)/ResultScreen', 
          params: { 
            result: JSON.stringify(apiResult), 
            image: imageUri 
          } 
        });
      }, 2000);
    } catch (error) {
      console.error('Identification error:', error);
      setProgressText('Error occurred during analysis');
      setLoadingState('error');
      setTimeout(() => {
        Alert.alert('Error', 'Failed to connect to server. Please try again.');
        router.back();
      }, 3000);
    }
  };

  const getStatusColor = () => {
    switch (loadingState) {
      case 'processing': return '#f59e0b';
      case 'completed': return '#10b981';
      case 'error': return '#ef4444';
      default: return '#f59e0b';
    }
  };

  const renderStatusIcon = () => {
    switch (loadingState) {
      case 'processing':
        return (
          <Animated.View style={{ transform: [{ rotate: spin }], marginRight: 8 }}>
            <Loader size={28} color={getStatusColor()} />
          </Animated.View>
        );
      case 'completed':
        return <CheckCircle size={28} color={getStatusColor()} style={{ marginRight: 8 }} />;
      case 'error':
        return <XCircle size={28} color={getStatusColor()} style={{ marginRight: 8 }} />;
      default:
        return null;
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center' }}>
          {renderStatusIcon()}
          <Text style={[styles.statusText, { color: getStatusColor(), marginRight: 0 }]}> 
            {progressText}
          </Text>
        </View>
        {loadingState === 'processing' && (
          <Image 
            source={require('../../assets/catWalking.gif')} 
            style={styles.catGif} 
            resizeMode="contain" 
          />
        )}
        {loadingState === 'completed' && (
          <View style={styles.completedContainer}>
            <CheckCircle size={64} color={getStatusColor()} style={{ marginBottom: 16 }} />
            <Text style={styles.completedMessage}>Results ready!</Text>
            <Text style={styles.redirectText}>Redirecting to results...</Text>
          </View>
        )}
        {loadingState === 'error' && (
          <View style={styles.errorContainer}>
            <XCircle size={64} color={getStatusColor()} style={{ marginBottom: 16 }} />
            <Text style={styles.errorMessage}>Something went wrong</Text>
          </View>
        )}
        {loadingState === 'processing' && (
          <ActivityIndicator 
            size="large" 
            color={getStatusColor()} 
            style={styles.spinner}
          />
        )}
      </View>
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
  content: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  statusText: {
    fontSize: 18,
    marginBottom: 32,
    marginTop: 32,
    fontWeight: '500',
    textAlign: 'center',
  },
  catGif: {
    width: 180,
    height: 180,
    marginBottom: 24,
  },
  spinner: {
    marginTop: 24,
  },
  completedContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  completedText: {
    fontSize: 64,
    marginBottom: 16,
  },
  completedMessage: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#10b981',
    marginBottom: 8,
  },
  redirectText: {
    fontSize: 16,
    color: '#6b7280',
  },
  errorContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  errorText: {
    fontSize: 64,
    marginBottom: 16,
  },
  errorMessage: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ef4444',
  },
});

export default LoadingScreen; 