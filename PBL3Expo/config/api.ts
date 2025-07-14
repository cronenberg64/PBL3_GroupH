// API Configuration for Cat Re-Identification System
// Updated for SavedModel backend compatibility

export const API_CONFIG = {
  // Backend server URL - update this to match your Mac's IP address
  BASE_URL: 'http://172.31.196.31:5002', // Updated to current Mac's IP
  
  // API endpoints
  ENDPOINTS: {
    IDENTIFY: '/identify',
    UPLOAD: '/upload',
    STATUS: '/status',
  },
  
  // Request timeout in milliseconds (30 seconds)
  TIMEOUT: 30000,
  
  // Model configuration
  MODEL_CONFIG: {
    INPUT_SHAPE: [150, 150, 3],
    EMBEDDING_DIM: 128,
    THRESHOLD: 0.4, // Similarity threshold for matching
  },
  
  // Error messages
  ERROR_MESSAGES: {
    NETWORK_ERROR: 'Network error. Please check your connection.',
    SERVER_ERROR: 'Server error. Please try again later.',
    TIMEOUT_ERROR: 'Request timed out. Please try again.',
    NO_CAT_DETECTED: 'No cat detected in the image. Please ensure the image contains a clear view of a cat.',
    NO_MATCH_FOUND: 'No matching cat found in our database.',
  }
};

// Helper function to get full API URL
export const getApiUrl = (endpoint: string): string => {
  return `${API_CONFIG.BASE_URL}${endpoint}`;
};

// Helper function to get identify endpoint URL
export const getIdentifyUrl = (): string => {
  return getApiUrl(API_CONFIG.ENDPOINTS.IDENTIFY);
};

// Helper function to get status endpoint URL
export const getStatusUrl = (): string => {
  return getApiUrl(API_CONFIG.ENDPOINTS.STATUS);
};

// Helper function to check if backend is available
export const checkBackendStatus = async (): Promise<boolean> => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    
    const response = await fetch(getStatusUrl(), {
      method: 'GET',
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    console.error('Backend status check failed:', error);
    return false;
  }
};

// Helper function to handle API errors
export const handleApiError = (error: any): string => {
  if (error.name === 'TypeError' && error.message.includes('fetch')) {
    return API_CONFIG.ERROR_MESSAGES.NETWORK_ERROR;
  }
  if (error.message?.includes('timeout')) {
    return API_CONFIG.ERROR_MESSAGES.TIMEOUT_ERROR;
  }
  if (error.status >= 500) {
    return API_CONFIG.ERROR_MESSAGES.SERVER_ERROR;
  }
  return error.message || 'An unexpected error occurred.';
}; 