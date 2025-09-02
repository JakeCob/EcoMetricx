# PDF Text Extractor Setup Guide

## Overview
This project uses `pymupdf4llm` to extract text and structural information from PDF documents, converting them to markdown format optimized for LLM applications and embedding creation.

## System Requirements
- Python 3.11 (recommended) or 3.10+
- Conda or Miniconda
- Minimum 1GB free disk space for output files

## Installation Steps

### 1. Create Conda Environment

```bash
# Create new conda environment with Python 3.11
conda create -n pdf-extractor python=3.11 -y

# Activate the environment
conda activate pdf-extractor
```

### 2. Install Dependencies

```bash
# Install pymupdf4llm (this will automatically install PyMuPDF as dependency)
pip install pymupdf4llm

# Optional: Install additional useful packages
pip install jupyter notebook pandas numpy
```

### 3. Verify Installation

```bash
python -c "import pymupdf4llm; print('✅ pymupdf4llm installed successfully')"
```

## Project Structure

```
pdf-extractor/
├── environment.yml          # Conda environment specification
├── pdf_extractor.py         # Main extraction script
├── test_extraction.py       # Test script
├── requirements.txt         # Python dependencies
├── README.md               # This documentation
└── output/                 # Generated output directory
    ├── images/             # Extracted images
    ├── extracted_text.md   # Extracted markdown
    └── page_chunks.json    # Page-by-page data
```

## Usage Examples

### Basic Text Extraction
```bash
# Extract all text from PDF
python pdf_extractor.py your_document.pdf

# Extract specific pages (0-based indexing)
python pdf_extractor.py your_document.pdf --pages 0 1 2

# Extract with custom output directory
python pdf_extractor.py your_document.pdf --output-dir /path/to/output
```

### Advanced Extraction with Images
```bash
# Extract text and images
python pdf_extractor.py your_document.pdf --with-images

# Extract with custom image format and resolution
python pdf_extractor.py your_document.pdf --with-images --image-format jpg --dpi 300
```

### Programmatic Usage
```python
from pdf_extractor import PDFTextExtractor

# Initialize extractor
extractor = PDFTextExtractor("output")

# Extract markdown text
md_text = extractor.extract_text_to_markdown("document.pdf")

# Extract with images
page_chunks = extractor.extract_with_images("document.pdf")

# Save results
extractor.save_markdown(md_text)
extractor.save_page_chunks(page_chunks)

# Get statistics
stats = extractor.get_document_stats(md_text)
print(stats)
```

## Output Formats

### 1. Markdown Text (.md)
- Preserves document structure with headers
- Maintains tables in markdown format
- Includes formatted text (bold, italic, lists)
- Optimized for LLM consumption and embedding creation

### 2. Page Chunks (JSON)
- Page-by-page breakdown
- Metadata for each page
- Image references
- Ideal for chunked processing and RAG systems

### 3. Extracted Images
- High-resolution images from PDF
- Named with page and sequence numbers
- Configurable format and DPI

## Why pymupdf4llm for Embeddings?

1. **Structured Output**: Maintains semantic structure crucial for meaningful embeddings
2. **Fast Processing**: Rule-based extraction is much faster than LLM-based alternatives
3. **Consistent Results**: Deterministic output perfect for batch processing
4. **LLM-Optimized**: Markdown format is ideal for embedding models
5. **Cost-Effective**: No API costs or rate limits

## Common Use Cases

- **RAG Systems**: Convert PDFs to searchable knowledge bases
- **Document Analysis**: Extract structured data for analysis
- **Embedding Creation**: Prepare text for vector embeddings
- **Content Migration**: Convert legacy PDFs to modern formats

## Troubleshooting

### ImportError: No module named 'pymupdf4llm'
```bash
pip install pymupdf4llm
```

### Permission denied errors
```bash
chmod +x pdf_extractor.py
```

### Large PDF processing
For very large PDFs, consider:
- Processing specific pages only
- Using lower DPI for images
- Processing in batches

### Memory issues
```bash
# Increase Python memory limit if needed
export PYTHONHASHSEED=0
ulimit -v 8000000  # Set memory limit
```

## Testing Your Installation

Run the included test script:
```bash
python test_extraction.py
```

This will validate that all components are working correctly.

## Environment Variables

Optional environment variables for customization:
```bash
export PDF_EXTRACTOR_OUTPUT_DIR="./custom_output"
export PDF_EXTRACTOR_DEFAULT_DPI="200"
export PDF_EXTRACTOR_LOG_LEVEL="DEBUG"
```

## Performance Notes

- **Small PDFs** (< 10 pages): ~1-2 seconds processing time
- **Medium PDFs** (10-50 pages): ~5-15 seconds processing time  
- **Large PDFs** (50+ pages): ~30+ seconds processing time
- **With images**: Add 2-5x processing time depending on image count and DPI

## Next Steps

After extraction, your markdown text is ready for:
1. **Embedding creation** using sentence-transformers or OpenAI embeddings
2. **Vector database ingestion** (Pinecone, Chroma, FAISS)
3. **RAG system integration** with LangChain or LlamaIndex
4. **Further text processing** and analysis

## Support

For issues with:
- **pymupdf4llm**: Check [PyMuPDF4LLM documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- **This script**: Review the error logs and ensure all dependencies are installed
- **PDF parsing problems**: Try processing individual pages to isolate issues

# 🚀 Quick Start Guide - PDF Text Extractor

## Installation (5 minutes)

### Option 1: Using Conda (Recommended)
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate pdf-extractor

# Test installation
python test_extraction.py
```

### Option 2: Using pip
```bash
# Create virtual environment
python -m venv pdf-extractor
source pdf-extractor/bin/activate  # On Windows: pdf-extractor\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_extraction.py
```

## Basic Usage

### Extract text from your energy report PDF
```bash
# Simple extraction - creates output/extracted_text.md
python pdf_extractor.py energy_report.pdf

# Extract with images - creates output/page_chunks.json and images/
python pdf_extractor.py energy_report.pdf --with-images
```

### In Python code
```python
from pdf_extractor import PDFTextExtractor

extractor = PDFTextExtractor()
md_text = extractor.extract_text_to_markdown("energy_report.pdf")
print(f"Extracted {len(md_text)} characters")

# Save for embedding creation
extractor.save_markdown(md_text, "energy_report.md")
```

## Expected Output

### From your energy report PDF:
- **Text**: ~2,235 characters, 368 words
- **Structure**: 12 headers, 5 table lines  
- **Content**: Account info, usage data, energy tips
- **Format**: Clean markdown perfect for embeddings

### Output Files:
```
output/
├── extracted_text.md      # Clean markdown text
├── page_chunks.json       # Page-by-page data
└── images/               # Extracted charts/graphics
    ├── energy_report-0-0.png
    └── energy_report-1-0.png
```

## Embedding Creation Ready

The extracted markdown is optimized for:
- ✅ **Vector embeddings** (sentence-transformers, OpenAI)
- ✅ **RAG systems** (LangChain, LlamaIndex)  
- ✅ **Chunked processing** (automatic chunking support)
- ✅ **LLM fine-tuning** (structured text format)

## Troubleshooting

### Common issues:
```bash
# Module not found
pip install pymupdf4llm

# Permission denied  
chmod +x pdf_extractor.py

# Test your setup
python -c "import pymupdf4llm; print('✅ Ready to go!')"
```

## Next Steps

1. **Extract your PDF**: `python pdf_extractor.py your_file.pdf`
2. **Create embeddings** from the markdown output
3. **Build your RAG system** using the structured text
4. **Query your documents** with an LLM

**Perfect for your energy report analysis task!** 🎯