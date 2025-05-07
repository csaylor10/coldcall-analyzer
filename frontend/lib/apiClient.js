import axios from 'axios';

let baseURL = '';

if (typeof window !== 'undefined') {
  // Client-side only: get base URL from window
  baseURL = `${window.location.origin}`;
} else {
  // Server-side: optionally use a fallback like env or default
  baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000';
}

const apiClient = axios.create({
  baseURL,
  headers: {
    'Accept': 'application/json',
  },
});

export default apiClient;
