// EcoMetricx API Service
import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { SearchRequest, SearchResponse, Document, SystemStatus } from '../types';

class EcoMetricxAPI {
  private client: AxiosInstance;

  constructor(baseURL: string = 'http://localhost:8000', apiKey?: string) {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        ...(apiKey && { 'X-API-Key': apiKey }),
      },
    });

    // Request interceptor for loading states
    this.client.interceptors.request.use((config) => {
      console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
      return config;
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error);
        if (error.response?.status === 401) {
          // Handle unauthorized
          console.error('Unauthorized access - check API key');
        }
        return Promise.reject(error);
      }
    );
  }

  // Search documents
  async search(request: SearchRequest): Promise<SearchResponse> {
    try {
      const response: AxiosResponse<SearchResponse> = await this.client.post('/search', request);
      return response.data;
    } catch (error) {
      console.error('Search failed:', error);
      throw new Error('Failed to search documents');
    }
  }

  // Find similar documents
  async findSimilar(chunkId: string, k: number = 5): Promise<SearchResponse> {
    try {
      const response: AxiosResponse<SearchResponse> = await this.client.post('/similar', {
        chunk_id: chunkId,
        k,
      });
      return response.data;
    } catch (error) {
      console.error('Similar search failed:', error);
      throw new Error('Failed to find similar documents');
    }
  }

  // Get system status
  async getSystemStatus(): Promise<SystemStatus> {
    try {
      const response: AxiosResponse<SystemStatus> = await this.client.get('/debug/config');
      return {
        ...response.data,
        status: response.data.status || 'healthy',
      };
    } catch (error) {
      console.error('System status check failed:', error);
      return {
        status: 'error',
      };
    }
  }

  // Upload and process document (placeholder for future implementation)
  async uploadDocument(file: File, onProgress?: (progress: number) => void): Promise<Document> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await this.client.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total && onProgress) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            onProgress(progress);
          }
        },
      });

      return response.data;
    } catch (error) {
      console.error('Upload failed:', error);
      throw new Error('Failed to upload document');
    }
  }

  // Get document details
  async getDocument(documentId: string): Promise<Document> {
    try {
      const response: AxiosResponse<Document> = await this.client.get(`/documents/${documentId}`);
      return response.data;
    } catch (error) {
      console.error('Get document failed:', error);
      throw new Error('Failed to get document details');
    }
  }

  // List documents
  async listDocuments(limit: number = 50, offset: number = 0): Promise<Document[]> {
    try {
      const response: AxiosResponse<{ documents: Document[] }> = await this.client.get('/documents', {
        params: { limit, offset },
      });
      return response.data.documents || [];
    } catch (error) {
      console.error('List documents failed:', error);
      throw new Error('Failed to list documents');
    }
  }

  // Update API key
  setApiKey(apiKey: string) {
    this.client.defaults.headers['X-API-Key'] = apiKey;
  }

  // Update base URL
  setBaseURL(baseURL: string) {
    this.client.defaults.baseURL = baseURL;
  }
}

// Create and export API instance
export const ecometricxAPI = new EcoMetricxAPI('http://localhost:8000', '-3h797xCB7IVJs9sBCfMA9rpttN7cTMZSdYtoYqpa0dFFbAJ2_gteQM5jfTPaWXf');
export default ecometricxAPI;
