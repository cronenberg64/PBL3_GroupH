import { decode as atob } from 'base-64';

// Helper to convert base64 image to Float32Array [1, 3, 224, 224]
export async function base64ToFloat32Array(base64: string): Promise<Float32Array> {
  // Decode base64 to binary string
  const binary = atob(base64);
  // Convert binary string to Uint8Array
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  // Use canvas to decode image and get pixel data (not available in React Native)
  // Instead, use expo-image-manipulator to get base64, then use a library like jpeg-js or png-js to decode
  // For now, return a dummy tensor (all zeros) for placeholder
  // TODO: Implement actual decoding using a library or native module
  return new Float32Array(1 * 3 * 224 * 224);
} 