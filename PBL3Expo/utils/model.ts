// Returns the mean RGB embedding from a 224x224 RGB image array (Uint8ClampedArray or number[])
export function getEmbedding(imageData: Uint8ClampedArray | number[]): number[] {
  let r = 0, g = 0, b = 0;
  const n = 224 * 224;
  for (let i = 0; i < n; i++) {
    r += imageData[i * 3];
    g += imageData[i * 3 + 1];
    b += imageData[i * 3 + 2];
  }
  return [r / n / 255, g / n / 255, b / n / 255];
} 