// API Configuration
export const API_CONFIG = {
  BASE_URL: 'http://192.168.40.138:5002',
  
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