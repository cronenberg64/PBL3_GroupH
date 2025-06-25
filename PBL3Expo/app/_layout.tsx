import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import { useFonts } from 'expo-font';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import 'react-native-reanimated';

import { useColorScheme } from '@/hooks/useColorScheme';

export default function RootLayout() {
  const colorScheme = useColorScheme();
  const [loaded] = useFonts({
    SpaceMono: require('../assets/fonts/SpaceMono-Regular.ttf'),
  });

  if (!loaded) {
    // Async font loading only occurs in development.
    return null;
  }

  return (
    <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
      <Stack>
        {/* Authentication Flow */}
        <Stack.Screen name="LoginScreen" options={{ headerShown: false, presentation: 'modal' }} />
        <Stack.Screen name="RegisterScreen" options={{ headerShown: false, presentation: 'modal' }} />
        {/* Main App Tabs */}
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        {/* Detail/Other Screens (hidden from tab bar) */}
        <Stack.Screen name="HomeScreen" options={{ headerShown: false, presentation: 'card' }} />
        <Stack.Screen name="ProfileScreen" options={{ headerShown: false, presentation: 'card' }} />
        <Stack.Screen name="NotFoundScreen" options={{ headerShown: false, presentation: 'card' }} />
        <Stack.Screen name="CatDetailScreen" options={{ headerShown: false, presentation: 'card' }} />
        <Stack.Screen name="MedicalHistoryScreen" options={{ headerShown: false, presentation: 'card' }} />
        {/* Fallback */}
        <Stack.Screen name="+not-found" />
      </Stack>
      <StatusBar style="auto" />
    </ThemeProvider>
  );
}
