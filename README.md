# EcoMetricx - AI-Powered Document Intelligence Platform

**Advanced PDF processing and intelligent search system with modern web interface**

---

## 🌟 **Overview**

EcoMetricx is a comprehensive document intelligence platform that transforms complex PDFs into searchable, structured data using advanced AI techniques. Built with a clean, modular architecture optimized for modern web development.

### ✨ **Key Features**

- 🔍 **Multi-Modal PDF Processing**: Advanced text extraction, visual element detection, and image analysis
- 🧠 **AI-Powered Search**: Vector embeddings with BGE models and hybrid search capabilities
- 📊 **Visual Intelligence**: Automatic detection and extraction of tables, charts, and images  
- 🌐 **REST API**: Production-ready FastAPI service for integration
- 📱 **Modern UI**: Clean, responsive interface for document management and search
- 🗄️ **Database Integration**: PostgreSQL with pgvector for scalable vector search

---

## 🚀 **Quick Start**

### **1. Environment Setup**
```bash
# Clone repository
git clone <repository-url>
cd EcoMetricx

# Create conda environment
conda env create -f environment.yml
conda activate pdf-extractor

# Install dependencies
pip install -r requirements.txt
```

### **2. Try the Complete Pipeline**
```bash
# Run the complete demonstration notebook
jupyter notebook 01_complete_ecometricx_pipeline.ipynb
```

### **3. Start the API Server**
```bash
# Configure environment
cp .env.example .env  # Edit with your settings

# Start the FastAPI service
uvicorn services.retrieval_api.main:app --reload --host 0.0.0.0 --port 8000
```

### **4. Launch the Web Interface**
```bash
# Start the UI (coming soon)
cd ui/frontend
npm start
```

---

## 🏗️ **Architecture**

### **Core Components**

```
EcoMetricx/
├── 📄 01_complete_ecometricx_pipeline.ipynb  # Complete demo & tutorial
├── 📁 src/                                    # Core processing engine
│   ├── extractors/    # PDF & document processors
│   ├── core/          # Utilities & output management  
│   └── scripts/       # Processing pipelines
├── 📁 services/       # API services
│   └── retrieval_api/ # FastAPI search service
├── 📁 ui/            # Web interface (React/Vue)
├── 📁 docs/          # Technical documentation
└── 📁 dev/           # Development tools
```

### **Processing Pipeline**

```mermaid
graph LR
    A[📄 PDF Upload] --> B[🔍 Text Extraction]
    B --> C[👁️ Visual Analysis] 
    C --> D[🧠 AI Enhancement]
    D --> E[🗄️ Vector Storage]
    E --> F[🔍 Search & Query]
    F --> G[📱 Web Interface]
```

---

## 💡 **Use Cases**

### **📊 Energy Reports**
- Extract usage data, bills, and efficiency metrics
- Analyze charts and graphs automatically
- Compare historical energy consumption

### **📋 Business Documents**
- Process contracts, invoices, and reports
- Extract structured data and key information
- Enable intelligent document search

### **🔬 Research Papers**
- Extract figures, tables, and citations
- Build searchable knowledge bases
- Cross-reference related content

---

## 🛠️ **Development**

### **Core Processing**
```python
from src.extractors.enhanced_pdf_extractor import EnhancedPDFTextExtractor
from src.extractors.visual_element_extractor import IntegratedVisualProcessor

# Extract text and structure
pdf_extractor = EnhancedPDFTextExtractor()
result = pdf_extractor.extract_with_layout_analysis("document.pdf")

# Extract visual elements
visual_extractor = IntegratedVisualProcessor()
elements = visual_extractor.process_pdf_page_visual("document.pdf")
```

### **API Integration**
```python
import requests

# Search documents
response = requests.post("http://localhost:8000/search", 
                        json={"query": "energy efficiency", "k": 5},
                        headers={"X-API-Key": "your-api-key"})
results = response.json()
```

### **Available APIs**
- `POST /search` - Intelligent document search with embeddings
- `POST /similar` - Find documents similar to a given one
- `GET /debug/config` - System status and configuration
- `POST /upload` - Upload and process new documents *(coming soon)*

---

## 🧪 **Testing**

```bash
# Run unit tests
pytest tests/

# Test API endpoints
python dev/test_api_queries.py

# API diagnostics
python dev/api_diagnostics.py
```

---

## 📊 **Performance**

| Feature | Performance | Notes |
|---------|-------------|-------|
| **Text Extraction** | ~0.3s per page | Enhanced PDF processing |
| **Visual Analysis** | ~2-4s per page | Computer vision pipeline |
| **Search Query** | <100ms | Vector similarity search |
| **API Response** | <200ms | Including text + vector fusion |

---

## 🤝 **Contributing**

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📚 **Documentation**

- 📖 **[Complete Pipeline Guide](01_complete_ecometricx_pipeline.ipynb)** - End-to-end tutorial
- 🏗️ **[Project Structure](docs/07_project_structure.md)** - Architecture overview
- 🔍 **[Search Improvements](docs/06_search_improvements.md)** - Latest enhancements
- 📊 **[Data Architecture](docs/01-data-architecture.md)** - Database design
- ⚙️ **[Processing Pipeline](docs/03-processing-pipeline.md)** - Technical details

---

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/ecometricx
POSTGRES_DSN=postgresql://user:pass@localhost/ecometricx

# Vector Database  
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-qdrant-key

# API Configuration
API_KEY=your-secure-api-key
FUSION_ALPHA=0.6  # Text vs vector search weight
ENABLE_RERANKER=false
```

### **Optional Dependencies**
- **PostgreSQL + pgvector**: For production database
- **Qdrant**: For vector search (alternative to pgvector)
- **Redis**: For caching (recommended for production)

---

## 🎯 **Roadmap**

### **🚧 In Development**
- [ ] **React Web Interface** - Modern, responsive UI
- [ ] **Batch Processing** - Handle multiple documents
- [ ] **Advanced Analytics** - Processing insights and metrics

### **🔮 Future Features**
- [ ] **Multi-language Support** - Process documents in various languages
- [ ] **Real-time Processing** - WebSocket-based live updates
- [ ] **Plugin System** - Custom extractors and processors
- [ ] **Mobile App** - iOS/Android companion app

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **FastAPI** - Modern, fast web framework for building APIs
- **BGE Embeddings** - State-of-the-art text embeddings
- **PyMuPDF** - Powerful PDF processing library
- **OpenCV** - Computer vision capabilities
- **React/Vue** - Modern frontend frameworks

---

**Built with ❤️ for the future of document intelligence**