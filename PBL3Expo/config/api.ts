// API Configuration
export const API_CONFIG = {
  // Change this to your computer's IP address on the same WiFi network
  // You can find your IP by running 'ifconfig' on Mac/Linux or 'ipconfig' on Windows
  BASE_URL: 'http://192.168.1.100:5000',
  
  // API endpoints
  ENDPOINTS: {
    IDENTIFY: '/identify',
    UPLOAD: '/upload',
  },
  
  // Request timeout in milliseconds
  TIMEOUT: 30000,
};

// Helper function to get full API URL
export const getApiUrl = (endpoint: string): string => {
  return `${API_CONFIG.BASE_URL}${endpoint}`;
};

// Helper function to get identify endpoint URL
export const getIdentifyUrl = (): string => {
  return getApiUrl(API_CONFIG.ENDPOINTS.IDENTIFY);
}; 