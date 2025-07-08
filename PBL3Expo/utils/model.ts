import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import * as FileSystem from 'expo-file-system';
import { Asset } from 'expo-asset';
import { Platform } from 'react-native';

let session: InferenceSession | null = null;

export async function loadModel() {
  if (session) return session;
  // Load the ONNX model from the app bundle
  const modelAsset = Asset.fromModule(require('../assets/model/cat_identifier.onnx'));
  await modelAsset.downloadAsync();
  let modelUri = modelAsset.localUri || modelAsset.uri;
  if (Platform.OS === 'android' && modelUri.startsWith('file://')) {
    modelUri = modelUri.replace('file://', '');
  }
  session = await InferenceSession.create(modelUri);
  return session;
}

// imageTensor: Float32Array or Tensor of shape [1, 3, 224, 224]
export async function getEmbedding(imageTensor: Float32Array) {
  const sess = await loadModel();
  const input = new Tensor('float32', imageTensor, [1, 3, 224, 224]);
  const output = await sess.run({ input });
  // Assume output is the first key
  const embedding = Object.values(output)[0].data as Float32Array;
  return embedding;
} 