import React, { useState } from 'react';
import { Input, Button, Card, List, Spin, Alert, Tag, Typography, Space, Divider } from 'antd';
import { SearchOutlined, FileTextOutlined, ClockCircleOutlined } from '@ant-design/icons';
import { SearchRequest, SearchResponse, SearchResult } from '../types';
import ecometricxAPI from '../services/api';

const { Search } = Input;
const { Text, Title } = Typography;

interface SearchInterfaceProps {
  onResultSelect?: (result: SearchResult) => void;
}

const SearchInterface: React.FC<SearchInterfaceProps> = ({ onResultSelect }) => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [query, setQuery] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [searchStats, setSearchStats] = useState<{
    totalResults: number;
    processingTime?: number;
    query: string;
  } | null>(null);

  const handleSearch = async (searchQuery: string) => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    setError(null);
    
    try {
      const request: SearchRequest = {
        query: searchQuery,
        k: 10, // Get top 10 results
      };

      const response: SearchResponse = await ecometricxAPI.search(request);
      
      setResults(response.results);
      setSearchStats({
        totalResults: response.total_results || response.results.length,
        processingTime: response.processing_time,
        query: response.query,
      });
    } catch (err) {
      setError('Failed to search documents. Please check your connection and try again.');
      console.error('Search error:', err);
    } finally {
      setLoading(false);
    }
  };

  const renderSearchResult = (result: SearchResult) => (
    <List.Item
      key={result.chunk_id}
      onClick={() => onResultSelect?.(result)}
      style={{ cursor: onResultSelect ? 'pointer' : 'default' }}
      className={onResultSelect ? 'search-result-item' : ''}
    >
      <List.Item.Meta
        avatar={<FileTextOutlined style={{ fontSize: '24px', color: '#1890ff' }} />}
        title={
          <Space>
            <Text strong>Document: {result.document_id}</Text>
            <Tag color="blue">Page {result.page_num}</Tag>
            <Tag color="green">Score: {(result.score * 100).toFixed(1)}%</Tag>
          </Space>
        }
        description={
          <div>
            <Text>{result.snippet}</Text>
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                Chunk ID: {result.chunk_id}
              </Text>
            </div>
          </div>
        }
      />
    </List.Item>
  );

  return (
    <div className="search-interface">
      <Card title={<Title level={3}><SearchOutlined /> Document Search</Title>}>
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <Search
            placeholder="Search documents (e.g., 'energy savings tips', 'monthly report')"
            allowClear
            enterButton={<Button type="primary" icon={<SearchOutlined />}>Search</Button>}
            size="large"
            onSearch={handleSearch}
            onChange={(e) => setQuery(e.target.value)}
            loading={loading}
          />

          {error && (
            <Alert
              message="Search Error"
              description={error}
              type="error"
              showIcon
              closable
              onClose={() => setError(null)}
            />
          )}

          {searchStats && (
            <Card size="small" style={{ backgroundColor: '#f6f8ff' }}>
              <Space split={<Divider type="vertical" />}>
                <Text>
                  <strong>{searchStats.totalResults}</strong> results for "{searchStats.query}"
                </Text>
                {searchStats.processingTime && (
                  <Text type="secondary">
                    <ClockCircleOutlined /> {(searchStats.processingTime * 1000).toFixed(0)}ms
                  </Text>
                )}
              </Space>
            </Card>
          )}

          {loading ? (
            <div style={{ textAlign: 'center', padding: '40px' }}>
              <Spin size="large" />
              <div style={{ marginTop: '16px' }}>
                <Text>Searching documents...</Text>
              </div>
            </div>
          ) : results.length > 0 ? (
            <List
              dataSource={results}
              renderItem={renderSearchResult}
              pagination={{
                pageSize: 5,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => 
                  `${range[0]}-${range[1]} of ${total} results`,
              }}
            />
          ) : query && !loading && (
            <div style={{ textAlign: 'center', padding: '40px' }}>
              <Text type="secondary">No results found. Try different keywords.</Text>
            </div>
          )}
        </Space>
      </Card>


    </div>
  );
};

export default SearchInterface;
