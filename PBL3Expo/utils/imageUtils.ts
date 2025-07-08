import { decode as atob } from 'base-64';
import jpeg from 'jpeg-js';
import * as ImageManipulator from 'expo-image-manipulator';

// Helper to convert base64 image to Float32Array [1, 3, 224, 224]
export async function base64ToFloat32Array(base64: string): Promise<Float32Array> {
  const binary = atob(base64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  const { width, height, data } = jpeg.decode(bytes, { useTArray: true });
  if (width !== 224 || height !== 224) {
    throw new Error('Image must be 224x224');
  }
  const floatArray = new Float32Array(1 * 3 * 224 * 224);
  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {
      const pixelIdx = (y * 224 + x) * 4;
      floatArray[0 * 224 * 224 + y * 224 + x] = data[pixelIdx] / 255;     // R
      floatArray[1 * 224 * 224 + y * 224 + x] = data[pixelIdx + 1] / 255; // G
      floatArray[2 * 224 * 224 + y * 224 + x] = data[pixelIdx + 2] / 255; // B
    }
  }
  return floatArray;
}

// Main preprocessing function: takes an image URI, resizes, converts to JPEG, and returns base64
export async function preprocessImageForEmbedding(uri: string): Promise<string> {
  const manipResult = await ImageManipulator.manipulateAsync(
    uri,
    [
      { resize: { width: 224, height: 224 } },
    ],
    { compress: 1, format: ImageManipulator.SaveFormat.JPEG, base64: true }
  );
  if (!manipResult.base64) {
    throw new Error('Failed to get base64 from image manipulator');
  }
  return manipResult.base64;
}

// Extracts RGB data as a flat array [R, G, B, R, G, B, ...] from a 224x224 JPEG base64 image
export function getImageRGBData(base64: string): number[] {
  const binary = atob(base64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  const { width, height, data } = jpeg.decode(bytes, { useTArray: true });
  if (width !== 224 || height !== 224) {
    throw new Error('Image must be 224x224');
  }
  const rgbArray: number[] = [];
  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {
      const pixelIdx = (y * 224 + x) * 4;
      rgbArray.push(data[pixelIdx], data[pixelIdx + 1], data[pixelIdx + 2]);
    }
  }
  return rgbArray;
} 