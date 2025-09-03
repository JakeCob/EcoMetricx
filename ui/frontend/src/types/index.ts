// EcoMetricx Frontend Types

export interface SearchResult {
  chunk_id: string;
  document_id: string;
  page_num: number;
  score: number;
  snippet: string;
}

export interface SearchRequest {
  query: string;
  k?: number;
  filter_document_id?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  query: string;
  total_results: number;
  processing_time?: number;
}

export interface Document {
  document_id: string;
  title?: string;
  filename?: string;
  pages?: number;
  upload_date?: string;
  file_size?: number;
  processing_status?: 'pending' | 'processing' | 'completed' | 'error';
}

export interface ExtractionResult {
  full_text: string;
  processing_time: number;
  total_pages: number;
  method: string;
  confidence?: number;
  layout_analysis?: any[];
  structured_data?: Record<string, any>;
}

export interface VisualElement {
  type: 'table' | 'chart' | 'image';
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  page_num: number;
  extracted_data?: any;
}

export interface UploadProgress {
  file: File;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  result?: ExtractionResult;
  error?: string;
}

export interface ApiConfig {
  baseUrl: string;
  apiKey: string;
}

export interface SystemStatus {
  status: 'healthy' | 'degraded' | 'error';
  version?: string;
  uptime?: string;
  documents_processed?: number;
  database_status?: string;
  vector_search_status?: string;
}
