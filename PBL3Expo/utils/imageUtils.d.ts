export function base64ToFloat32Array(base64: string): Promise<Float32Array>;
export function preprocessImageForEmbedding(uri: string): Promise<string>;
export function getImageRGBData(base64: string): number[]; 