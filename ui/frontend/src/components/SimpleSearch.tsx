import React, { useState } from 'react';
import { SearchResult } from '../types';
import ecometricxAPI from '../services/api';

const SimpleSearch: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      const response = await ecometricxAPI.search({
        query: query.trim(),
        k: 10,
      });
      setResults(response.results);
    } catch (err) {
      setError('Search failed. Please try again.');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="simple-search">
      <div className="search-container">
        <h1>EcoMetricx Document Search</h1>
        
        <form onSubmit={handleSearch} className="search-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter query"
            className="search-input"
            disabled={loading}
          />
          <button 
            type="submit" 
            className="search-button"
            disabled={loading || !query.trim()}
          >
            {loading ? 'Querying...' : 'Query'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {results.length > 0 && (
          <div className="results-section">
            <h2>Search Results ({results.length})</h2>
            <div className="results-list">
              {results.map((result, index) => (
                <div key={result.chunk_id} className="result-item">
                  <div className="result-header">
                    <span className="result-number">#{index + 1}</span>
                    <span className="document-info">
                      Document: {result.document_id} | Page: {result.page_num} | 
                      Score: {(result.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="result-content">
                    {result.snippet || 'No preview available'}
                  </div>
                  <div className="result-meta">
                    Chunk ID: {result.chunk_id}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {!loading && results.length === 0 && hasSearched && (
          <div className="no-results">
            No results found. Try different search terms.
          </div>
        )}
      </div>
    </div>
  );
};

export default SimpleSearch;