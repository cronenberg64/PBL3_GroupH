# Smart Cat Re-Identification (Client-Side Expo App)

This app is now fully client-side and offline-capable, using ONNX for AI inference directly on your device. No backend or server is required.

## Setup

1. **Place your ONNX model** in `assets/model/cat_identifier.onnx`.
2. **Install dependencies:**
   ```bash
   npm install
   npx expo install onnxruntime-react-native expo-image-picker expo-image-manipulator @react-native-async-storage/async-storage
   ```
3. **Run the app:**
   ```bash
   npx expo start
   ```

## How it works
- Select or capture a cat photo.
- The app runs the ONNX model locally to extract a feature embedding.
- Embeddings are compared to locally stored cat embeddings using cosine similarity.
- If a match is found (similarity > threshold), the cat is identified. Otherwise, you can register a new cat.

## Model requirements
- The ONNX model should accept a 224x224 RGB image and output a 1D embedding vector.
- You can convert your PyTorch model using `torch.onnx.export()`.

## Data storage
- All cat embeddings and metadata are stored locally using AsyncStorage.

## No backend required!
- All AI and data logic runs on-device. The app works offline and is compatible with Expo Go.

---

(For legacy backend instructions, see the old README in the project root.)
