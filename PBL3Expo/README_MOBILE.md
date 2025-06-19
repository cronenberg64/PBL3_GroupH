# Cat Identification Mobile App

This is the mobile application for the Cat Re-identification System built with Expo and React Native.

## Features

- 📸 **Camera Integration**: Take photos directly from the app
- 🖼️ **Gallery Picker**: Select images from your device's gallery
- 🤖 **AI Integration**: Connect to the Flask backend for cat identification
- 📱 **Cross-Platform**: Works on iOS, Android, and Web
- 🎨 **Modern UI**: Beautiful, intuitive interface with dark/light mode support

## Setup Instructions

### Prerequisites

1. **Node.js** (version 18 or higher)
2. **Expo CLI**: `npm install -g @expo/cli`
3. **Flask Backend**: Make sure the Python Flask server is running

### Installation

1. Navigate to the PBL3Expo directory:
   ```bash
   cd PBL3Expo
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Install additional required packages:
   ```bash
   npx expo install expo-camera expo-image-picker
   ```

### Configuration

1. **Update Server URL**: Edit `config/api.ts` and change the `BASE_URL` to your computer's IP address:
   ```typescript
   BASE_URL: 'http://YOUR_IP_ADDRESS:5000',
   ```

2. **Find Your IP Address**:
   - **Mac/Linux**: Run `ifconfig` in terminal
   - **Windows**: Run `ipconfig` in command prompt
   - Look for your local IP (usually starts with 192.168.x.x or 10.0.x.x)

### Running the App

1. **Start the development server**:
   ```bash
   npm start
   ```

2. **Run on different platforms**:
   - **iOS Simulator**: Press `i` in the terminal
   - **Android Emulator**: Press `a` in the terminal
   - **Web Browser**: Press `w` in the terminal
   - **Physical Device**: Scan the QR code with Expo Go app

### Using the App

1. **Take a Photo**: Tap "Take Photo" to use the camera
2. **Choose from Gallery**: Tap "Choose from Gallery" to select an existing image
3. **Identify Cat**: After selecting an image, tap "Identify Cat" to send it to the server
4. **View Results**: See the identification results with confidence scores

## Troubleshooting

### Common Issues

1. **"Cannot connect to server" error**:
   - Check that your Flask server is running (`python serve.py`)
   - Verify both devices are on the same WiFi network
   - Update the IP address in `config/api.ts`
   - Check firewall settings

2. **Camera permissions denied**:
   - Go to device settings and enable camera permissions for the app
   - On iOS, also check photo library permissions

3. **App crashes on startup**:
   - Clear Expo cache: `npx expo start --clear`
   - Reinstall dependencies: `rm -rf node_modules && npm install`

### Network Configuration

For the app to communicate with your Flask server:

1. **Same Network**: Both your computer and mobile device must be on the same WiFi network
2. **Correct IP**: Use your computer's local IP address, not localhost
3. **Port Access**: Ensure port 5000 is accessible (no firewall blocking)

### Development Tips

- Use Expo DevTools for debugging
- Enable hot reloading for faster development
- Test on both iOS and Android for compatibility
- Use the Expo Go app for quick testing on physical devices

## File Structure

```
PBL3Expo/
├── app/
│   └── (tabs)/
│       ├── index.tsx          # Main cat identification screen
│       ├── explore.tsx        # Information screen
│       └── _layout.tsx        # Tab navigation
├── config/
│   └── api.ts                 # API configuration
├── components/                # Reusable UI components
└── constants/
    └── Colors.ts             # Theme colors
```

## API Integration

The app communicates with the Flask backend through:

- **Endpoint**: `/identify`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Response**: JSON with identification results

Example response:
```json
{
  "match_found": true,
  "matched_id": "ryusei_1",
  "score": 0.05
}
```

## Next Steps

- [ ] Add user authentication
- [ ] Implement cat registration from mobile
- [ ] Add image preprocessing on device
- [ ] Implement offline mode
- [ ] Add push notifications
- [ ] Create admin panel for managing cat database 