#!/usr/bin/env python3
"""
Enhanced PDF Text Extractor with Advanced Layout Analysis

This script provides comprehensive PDF text extraction with:
- Advanced layout handling and structure preservation
- Image-text correlation and cross-referencing
- Structured data extraction with pattern recognition
- Embedding-optimized chunking strategies
- Multi-format output generation

Features:
- Multi-column layout detection and reading order optimization
- Advanced table extraction with proper structure preservation
- Header/footer section identification and handling
- Semantic chunking for optimal embedding creation
- Cross-reference generation with image locations
- Structured data pattern extraction (accounts, dates, figures)

Author: Assistant
Date: 2025-09-02
"""

import os
import sys
import json
import pathlib
import argparse
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import math

try:
    import pymupdf4llm
    import pymupdf as fitz
except ImportError:
    print("Error: PyMuPDF libraries not installed.")
    print("Please install using: pip install pymupdf4llm PyMuPDF")
    sys.exit(1)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Some advanced features will be disabled.")


class LayoutAnalyzer:
    """Analyzes PDF layout for better structure understanding."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_page_layout(self, page: fitz.Page) -> Dict[str, Any]:
        """
        Analyze page layout to understand structure.
        
        Args:
            page (fitz.Page): PDF page object
        
        Returns:
            Dict: Layout analysis results
        """
        page_rect = page.rect
        text_blocks = page.get_text("dict")
        
        layout_info = {
            "page_number": page.number,
            "page_size": {"width": page_rect.width, "height": page_rect.height},
            "columns": self._detect_columns(text_blocks, page_rect),
            "headers": self._detect_headers(text_blocks, page_rect),
            "footers": self._detect_footers(text_blocks, page_rect),
            "tables": self._detect_tables(page),
            "reading_order": self._determine_reading_order(text_blocks, page_rect)
        }
        
        return layout_info
    
    def _detect_columns(self, text_blocks: Dict, page_rect: fitz.Rect) -> List[Dict]:
        """Detect column structure on the page."""
        blocks = text_blocks.get("blocks", [])
        if not blocks:
            return [{"x0": 0, "x1": page_rect.width, "type": "single"}]
        
        # Analyze x-coordinates of text blocks
        x_positions = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        x_positions.extend([span["bbox"][0], span["bbox"][2]])
        
        if not x_positions:
            return [{"x0": 0, "x1": page_rect.width, "type": "single"}]
        
        # Simple column detection based on gaps
        x_positions = sorted(set(x_positions))
        gaps = []
        for i in range(len(x_positions) - 1):
            gap = x_positions[i + 1] - x_positions[i]
            if gap > 50:  # Significant gap threshold
                gaps.append((x_positions[i], x_positions[i + 1], gap))
        
        if not gaps:
            return [{"x0": 0, "x1": page_rect.width, "type": "single"}]
        
        # Create column boundaries
        columns = []
        current_x = 0
        for gap_start, gap_end, gap_size in gaps:
            if current_x < gap_start:
                columns.append({"x0": current_x, "x1": gap_start, "type": "text"})
            current_x = gap_end
        
        # Add final column
        if current_x < page_rect.width:
            columns.append({"x0": current_x, "x1": page_rect.width, "type": "text"})
        
        return columns if len(columns) > 1 else [{"x0": 0, "x1": page_rect.width, "type": "single"}]
    
    def _detect_headers(self, text_blocks: Dict, page_rect: fitz.Rect) -> List[Dict]:
        """Detect header sections."""
        headers = []
        blocks = text_blocks.get("blocks", [])
        
        header_threshold = page_rect.height * 0.15  # Top 15% of page
        
        for block in blocks:
            if "lines" in block and block["bbox"][1] <= header_threshold:
                headers.append({
                    "text": " ".join([span["text"] for line in block["lines"] 
                                   for span in line.get("spans", [])]),
                    "bbox": block["bbox"],
                    "confidence": 0.8
                })
        
        return headers
    
    def _detect_footers(self, text_blocks: Dict, page_rect: fitz.Rect) -> List[Dict]:
        """Detect footer sections."""
        footers = []
        blocks = text_blocks.get("blocks", [])
        
        footer_threshold = page_rect.height * 0.85  # Bottom 15% of page
        
        for block in blocks:
            if "lines" in block and block["bbox"][1] >= footer_threshold:
                footers.append({
                    "text": " ".join([span["text"] for line in block["lines"] 
                                   for span in line.get("spans", [])]),
                    "bbox": block["bbox"],
                    "confidence": 0.8
                })
        
        return footers
    
    def _detect_tables(self, page: fitz.Page) -> List[Dict]:
        """Detect table structures on the page."""
        tables = []
        
        try:
            # Try to find tables using PyMuPDF's table detection
            table_finder = page.find_tables()
            for table in table_finder:
                table_data = table.extract()
                if table_data and len(table_data) > 1:  # At least header + one row
                    tables.append({
                        "bbox": table.bbox,
                        "data": table_data,
                        "rows": len(table_data),
                        "cols": len(table_data[0]) if table_data else 0,
                        "confidence": 0.9
                    })
        except Exception as e:
            self.logger.debug(f"Table detection error: {str(e)}")
        
        return tables
    
    def _determine_reading_order(self, text_blocks: Dict, page_rect: fitz.Rect) -> List[int]:
        """Determine optimal reading order for text blocks."""
        blocks = text_blocks.get("blocks", [])
        if not blocks:
            return []
        
        # Sort blocks by position (top-to-bottom, left-to-right)
        block_positions = []
        for i, block in enumerate(blocks):
            if "lines" in block:
                bbox = block["bbox"]
                # Primary sort: top-to-bottom, Secondary: left-to-right
                sort_key = (bbox[1], bbox[0])  # (y0, x0)
                block_positions.append((sort_key, i))
        
        # Sort and return indices
        sorted_blocks = sorted(block_positions, key=lambda x: x[0])
        return [idx for _, idx in sorted_blocks]


class StructuredDataExtractor:
    """Extracts structured data patterns from PDF content."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Pattern definitions
        self.patterns = {
            "account_number": re.compile(r"Account\s+number:?\s*(\d+)", re.IGNORECASE),
            "service_address": re.compile(r"Service\s+address:?\s*([^\n\r]+)", re.IGNORECASE),
            "phone_number": re.compile(r"(\d{3}[.-]?\d{3}[.-]?\d{4})"),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "website": re.compile(r"(?:https?://)?(?:www\.)?([A-Za-z0-9.-]+\.[A-Za-z]{2,})", re.IGNORECASE),
            "percentage": re.compile(r"(\d+(?:\.\d+)?)\s*%"),
            "currency": re.compile(r"\$(\d+(?:,\d{3})*(?:\.\d{2})?)"),
            "energy_usage": re.compile(r"(\d+(?:,\d{3})*)\s*(?:kWh|kwh|KWH)", re.IGNORECASE),
            "date": re.compile(r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"),
        }
    
    def extract_structured_data(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data patterns from text.
        
        Args:
            text (str): Input text to analyze
        
        Returns:
            Dict: Extracted structured data
        """
        extracted_data = {}
        
        for pattern_name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                # Clean and deduplicate matches
                clean_matches = list(set([match.strip() if isinstance(match, str) else match 
                                        for match in matches]))
                extracted_data[pattern_name] = clean_matches
        
        # Post-process specific patterns
        extracted_data = self._post_process_data(extracted_data, text)
        
        return extracted_data
    
    def _post_process_data(self, data: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Post-process extracted data for better accuracy."""
        processed = data.copy()
        
        # Clean service addresses
        if "service_address" in processed:
            addresses = []
            for addr in processed["service_address"]:
                # Remove extra whitespace and common PDF artifacts
                clean_addr = re.sub(r'\s+', ' ', addr).strip()
                if len(clean_addr) > 5:  # Reasonable address length
                    addresses.append(clean_addr)
            processed["service_address"] = addresses
        
        # Validate and format energy usage
        if "energy_usage" in processed:
            usage_values = []
            for usage in processed["energy_usage"]:
                try:
                    # Remove commas and convert to integer
                    clean_usage = int(usage.replace(',', ''))
                    if 0 < clean_usage < 100000:  # Reasonable range
                        usage_values.append(clean_usage)
                except ValueError:
                    continue
            processed["energy_usage"] = usage_values
        
        # Extract customer name if present
        name_pattern = re.compile(r"Dear\s+([A-Z][A-Z\s]+),", re.IGNORECASE)
        name_match = name_pattern.search(text)
        if name_match:
            processed["customer_name"] = [name_match.group(1).strip()]
        
        return processed


class EmbeddingChunker:
    """Creates optimized text chunks for embedding generation."""
    
    def __init__(self, max_chunk_size: int = 512, overlap_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.logger = logging.getLogger(__name__)
    
    def create_chunks(self, 
                     text: str, 
                     strategy: str = "semantic",
                     metadata: Optional[Dict] = None,
                     image_references: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """
        Create optimized chunks for embeddings.
        
        Args:
            text (str): Input text to chunk
            strategy (str): Chunking strategy (semantic, size, structure, hybrid)
            metadata (Dict): Additional metadata for chunks
            image_references (List): Related image information
        
        Returns:
            List[Dict]: Optimized text chunks with metadata
        """
        if strategy == "semantic":
            return self._semantic_chunking(text, metadata, image_references)
        elif strategy == "size":
            return self._size_based_chunking(text, metadata, image_references)
        elif strategy == "structure":
            return self._structure_based_chunking(text, metadata, image_references)
        elif strategy == "hybrid":
            return self._hybrid_chunking(text, metadata, image_references)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def _semantic_chunking(self, text: str, metadata: Optional[Dict], 
                          image_references: Optional[List[Dict]]) -> List[Dict[str, Any]]:
        """Create chunks based on semantic boundaries."""
        chunks = []
        
        # Split by logical sections (headers, paragraphs, etc.)
        sections = re.split(r'\n\n+|\n#{1,6}\s', text)
        sections = [s.strip() for s in sections if s.strip()]
        
        current_chunk = ""
        chunk_metadata = metadata.copy() if metadata else {}
        
        for i, section in enumerate(sections):
            # Check if adding this section would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + section if current_chunk else section
            
            if len(potential_chunk) <= self.max_chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        current_chunk, len(chunks), chunk_metadata, image_references
                    ))
                
                # Start new chunk
                if len(section) <= self.max_chunk_size:
                    current_chunk = section
                else:
                    # Section too large, split it
                    sub_chunks = self._split_large_section(section)
                    for sub_chunk in sub_chunks[:-1]:
                        chunks.append(self._create_chunk_dict(
                            sub_chunk, len(chunks), chunk_metadata, image_references
                        ))
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                current_chunk, len(chunks), chunk_metadata, image_references
            ))
        
        return chunks
    
    def _size_based_chunking(self, text: str, metadata: Optional[Dict], 
                           image_references: Optional[List[Dict]]) -> List[Dict[str, Any]]:
        """Create fixed-size chunks with overlap."""
        chunks = []
        chunk_metadata = metadata.copy() if metadata else {}
        
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            
            # Try to end at word boundary
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(self._create_chunk_dict(
                    chunk_text, chunk_id, chunk_metadata, image_references
                ))
                chunk_id += 1
            
            # Move start position with overlap
            start = max(end - self.overlap_size, start + 1)
        
        return chunks
    
    def _structure_based_chunking(self, text: str, metadata: Optional[Dict], 
                                image_references: Optional[List[Dict]]) -> List[Dict[str, Any]]:
        """Create chunks preserving structural elements."""
        chunks = []
        chunk_metadata = metadata.copy() if metadata else {}
        
        # Identify structural elements
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a structural element (header, list item, etc.)
            is_header = re.match(r'^#{1,6}\s+', line)
            is_list_item = re.match(r'^[\*\-\+]\s+', line)
            
            line_size = len(line)
            
            # If adding this line would exceed limit and we have content
            if current_size + line_size > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(self._create_chunk_dict(
                        chunk_text, len(chunks), chunk_metadata, image_references
                    ))
                
                # Start new chunk
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(self._create_chunk_dict(
                    chunk_text, len(chunks), chunk_metadata, image_references
                ))
        
        return chunks
    
    def _hybrid_chunking(self, text: str, metadata: Optional[Dict], 
                        image_references: Optional[List[Dict]]) -> List[Dict[str, Any]]:
        """Combine multiple chunking strategies."""
        # Start with semantic chunking
        semantic_chunks = self._semantic_chunking(text, metadata, image_references)
        
        # Refine large chunks with size-based splitting
        refined_chunks = []
        for chunk in semantic_chunks:
            if len(chunk["text"]) > self.max_chunk_size * 1.5:
                # Split large semantic chunks
                sub_chunks = self._size_based_chunking(
                    chunk["text"], chunk["metadata"], chunk.get("image_references")
                )
                refined_chunks.extend(sub_chunks)
            else:
                refined_chunks.append(chunk)
        
        # Renumber chunks
        for i, chunk in enumerate(refined_chunks):
            chunk["chunk_id"] = i
        
        return refined_chunks
    
    def _split_large_section(self, section: str) -> List[str]:
        """Split a large section into smaller pieces."""
        sentences = re.split(r'[.!?]+\s+', section)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            potential_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.max_chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence[:self.max_chunk_size] if len(sentence) > self.max_chunk_size else sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [section[:self.max_chunk_size]]
    
    def _create_chunk_dict(self, text: str, chunk_id: int, metadata: Dict, 
                          image_references: Optional[List[Dict]]) -> Dict[str, Any]:
        """Create a standardized chunk dictionary."""
        chunk = {
            "chunk_id": chunk_id,
            "text": text,
            "character_count": len(text),
            "word_count": len(text.split()),
            "metadata": metadata.copy(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add image references if available
        if image_references:
            chunk["image_references"] = image_references
        
        # Calculate importance score
        chunk["importance_score"] = self._calculate_importance(text)
        
        # Identify content type
        chunk["content_type"] = self._identify_content_type(text)
        
        return chunk
    
    def _calculate_importance(self, text: str) -> float:
        """Calculate importance score for a text chunk."""
        score = 0.0
        
        # Header indicators
        if re.search(r'^#{1,6}\s', text, re.MULTILINE):
            score += 0.3
        
        # Contains structured data
        if re.search(r'\d+[%$]|\b\d{3,}\b', text):
            score += 0.2
        
        # Contains contact information
        if re.search(r'@|\.com|\d{3}[.-]?\d{3}[.-]?\d{4}', text):
            score += 0.2
        
        # Length factor (medium length preferred)
        length_score = min(len(text) / 200, 1.0) * 0.3
        score += length_score
        
        return min(score, 1.0)
    
    def _identify_content_type(self, text: str) -> str:
        """Identify the type of content in the chunk."""
        text_lower = text.lower()
        
        if re.search(r'^#{1,6}\s', text):
            return "header"
        elif "table" in text_lower or "|" in text:
            return "table"
        elif re.search(r'^[\*\-\+]\s', text, re.MULTILINE):
            return "list"
        elif any(word in text_lower for word in ["tip:", "advice:", "recommendation:"]):
            return "advice"
        elif any(word in text_lower for word in ["account", "service address", "number"]):
            return "account_info"
        elif re.search(r'\d+\s*%|\d+\s*kwh|\$\d+', text_lower):
            return "usage_data"
        elif "©" in text or "reserved" in text_lower:
            return "footer"
        else:
            return "body"


class EnhancedPDFTextExtractor:
    """
    Enhanced PDF text extractor with advanced layout analysis and integration capabilities.
    """
    
    def __init__(self, output_dir: str = "output", enable_structured_extraction: bool = True):
        """
        Initialize the Enhanced PDF Text Extractor.
        
        Args:
            output_dir (str): Directory to save extracted content
            enable_structured_extraction (bool): Enable structured data extraction
        """
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create enhanced directory structure
        self.text_dir = self.output_dir / "text"
        self.integrated_dir = self.output_dir / "integrated"
        self.embeddings_ready_dir = self.output_dir / "embeddings_ready"
        
        for dir_path in [self.text_dir, self.integrated_dir, self.embeddings_ready_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.layout_analyzer = LayoutAnalyzer()
        self.data_extractor = StructuredDataExtractor()
        self.chunker = EmbeddingChunker()
        self.enable_structured_extraction = enable_structured_extraction
    
    def extract_with_layout_analysis(self, pdf_path: str, preserve_structure: bool = True) -> Dict[str, Any]:
        """
        Extract text with advanced layout analysis.
        
        Args:
            pdf_path (str): Path to PDF file
            preserve_structure (bool): Whether to preserve original structure
        
        Returns:
            Dict: Enhanced extraction results
        """
        pdf_path = pathlib.Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Starting enhanced extraction from: {pdf_path}")
        
        try:
            # Open PDF document
            pdf_doc = fitz.open(str(pdf_path))
            
            if pdf_doc.is_encrypted:
                self.logger.warning("PDF is encrypted. Attempting to open...")
                if not pdf_doc.authenticate(""):
                    raise ValueError("PDF requires password authentication")
            
            extraction_results = {
                "source_pdf": pdf_path.name,
                "extraction_date": datetime.now().isoformat(),
                "total_pages": len(pdf_doc),
                "pages": [],
                "layout_analysis": [],
                "structured_data": {},
                "full_text": "",
                "embedding_chunks": []
            }
            
            full_text_parts = []
            
            # Process each page
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                
                # Analyze layout
                layout_info = self.layout_analyzer.analyze_page_layout(page)
                extraction_results["layout_analysis"].append(layout_info)
                
                # Extract text using pymupdf4llm for better formatting
                page_text = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_num])
                
                # Clean and enhance the extracted text
                cleaned_text = self._clean_extracted_text(page_text, layout_info)
                
                page_info = {
                    "page_number": page_num,
                    "text": cleaned_text,
                    "layout": layout_info,
                    "character_count": len(cleaned_text),
                    "word_count": len(cleaned_text.split())
                }
                
                extraction_results["pages"].append(page_info)
                full_text_parts.append(cleaned_text)
                
                self.logger.debug(f"Processed page {page_num + 1}/{len(pdf_doc)}")
            
            # Store page count before closing document
            total_pages = len(pdf_doc)
            pdf_doc.close()
            
            # Combine all text
            full_text = "\n\n".join(full_text_parts)
            extraction_results["full_text"] = full_text
            
            # Extract structured data
            if self.enable_structured_extraction:
                extraction_results["structured_data"] = self.data_extractor.extract_structured_data(full_text)
            
            self.logger.info(f"Successfully extracted text from {total_pages} pages")
            return extraction_results
            
        except Exception as e:
            self.logger.error(f"Error during enhanced extraction: {str(e)}")
            raise
    
    def extract_with_image_references(self, pdf_path: str, image_metadata_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract text with image location markers and cross-references.
        
        Args:
            pdf_path (str): Path to PDF file
            image_metadata_path (str): Path to image metadata JSON file
        
        Returns:
            Dict: Extraction results with image cross-references
        """
        # First do standard enhanced extraction
        results = self.extract_with_layout_analysis(pdf_path)
        
        # Load image metadata if provided
        image_metadata = {}
        if image_metadata_path and pathlib.Path(image_metadata_path).exists():
            try:
                with open(image_metadata_path, 'r') as f:
                    image_data = json.load(f)
                    for img in image_data.get("images", []):
                        page_num = img.get("page_number", 0)
                        if page_num not in image_metadata:
                            image_metadata[page_num] = []
                        image_metadata[page_num].append(img)
                self.logger.info(f"Loaded image metadata for {len(image_data.get('images', []))} images")
            except Exception as e:
                self.logger.warning(f"Could not load image metadata: {str(e)}")
        
        # Add image references to text
        if image_metadata:
            results = self._add_image_references(results, image_metadata)
        
        return results
    
    def create_embedding_chunks(self, text: str, chunk_strategy: str = "semantic", 
                              image_references: Optional[List[Dict]] = None,
                              max_chunk_size: int = 512) -> List[Dict[str, Any]]:
        """
        Create embedding-optimized text chunks.
        
        Args:
            text (str): Input text
            chunk_strategy (str): Chunking strategy
            image_references (List): Related image information
            max_chunk_size (int): Maximum chunk size
        
        Returns:
            List[Dict]: Optimized chunks
        """
        # Update chunker settings
        self.chunker.max_chunk_size = max_chunk_size
        
        metadata = {
            "chunk_strategy": chunk_strategy,
            "max_chunk_size": max_chunk_size,
            "creation_timestamp": datetime.now().isoformat()
        }
        
        chunks = self.chunker.create_chunks(text, chunk_strategy, metadata, image_references)
        
        self.logger.info(f"Created {len(chunks)} chunks using {chunk_strategy} strategy")
        return chunks
    
    def extract_tables_advanced(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Advanced table extraction with proper structure preservation.
        
        Args:
            pdf_path (str): Path to PDF file
        
        Returns:
            List[Dict]: Extracted tables with metadata
        """
        tables = []
        
        try:
            pdf_doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                page_tables = self.layout_analyzer._detect_tables(page)
                
                for i, table_info in enumerate(page_tables):
                    table_dict = {
                        "table_id": f"page{page_num}_table{i}",
                        "page_number": page_num,
                        "bbox": table_info["bbox"],
                        "data": table_info["data"],
                        "rows": table_info["rows"],
                        "cols": table_info["cols"],
                        "confidence": table_info["confidence"]
                    }
                    
                    # Convert to different formats
                    if PANDAS_AVAILABLE and table_info["data"]:
                        try:
                            df = pd.DataFrame(table_info["data"][1:], columns=table_info["data"][0])
                            table_dict["formats"] = {
                                "csv": df.to_csv(index=False),
                                "json": df.to_json(orient="records"),
                                "markdown": df.to_markdown(index=False)
                            }
                        except Exception as e:
                            self.logger.debug(f"Could not convert table to DataFrame: {str(e)}")
                    
                    tables.append(table_dict)
            
            pdf_doc.close()
            
        except Exception as e:
            self.logger.error(f"Error extracting tables: {str(e)}")
        
        return tables
    
    def save_enhanced_results(self, results: Dict[str, Any], base_filename: str = "enhanced_extraction") -> Dict[str, str]:
        """
        Save enhanced extraction results in multiple formats.
        
        Args:
            results (Dict): Extraction results
            base_filename (str): Base filename for output files
        
        Returns:
            Dict: Paths to saved files
        """
        saved_files = {}
        
        try:
            # Save full markdown text
            md_path = self.text_dir / f"{base_filename}.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(results.get("full_text", ""))
            saved_files["full_text_markdown"] = str(md_path)
            
            # Save structured data
            if results.get("structured_data"):
                structured_path = self.text_dir / f"{base_filename}_structured_data.json"
                with open(structured_path, 'w', encoding='utf-8') as f:
                    json.dump(results["structured_data"], f, indent=2, ensure_ascii=False)
                saved_files["structured_data"] = str(structured_path)
            
            # Save embedding chunks
            if results.get("embedding_chunks"):
                chunks_path = self.embeddings_ready_dir / f"{base_filename}_chunks.json"
                with open(chunks_path, 'w', encoding='utf-8') as f:
                    json.dump(results["embedding_chunks"], f, indent=2, ensure_ascii=False)
                saved_files["embedding_chunks"] = str(chunks_path)
            
            # Save complete extraction report
            report_path = self.integrated_dir / f"{base_filename}_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            saved_files["complete_report"] = str(report_path)
            
            # Save cross-references if available
            if results.get("cross_references"):
                cross_ref_path = self.text_dir / f"{base_filename}_cross_references.json"
                with open(cross_ref_path, 'w', encoding='utf-8') as f:
                    json.dump(results["cross_references"], f, indent=2, ensure_ascii=False)
                saved_files["cross_references"] = str(cross_ref_path)
            
            self.logger.info(f"Saved enhanced results to {len(saved_files)} files")
            
        except Exception as e:
            self.logger.error(f"Error saving enhanced results: {str(e)}")
            raise
        
        return saved_files
    
    def _clean_extracted_text(self, text: str, layout_info: Dict[str, Any]) -> str:
        """Clean and enhance extracted text based on layout analysis."""
        if not text:
            return ""
        
        # Remove excessive line breaks
        text = re.sub(r'\n\n\n+', '\n\n', text)
        
        # Fix broken table formatting (remove Col2, Col3, etc.)
        text = re.sub(r'\|Col\d+', '|', text)
        text = re.sub(r'Col\d+\|', '|', text)
        
        # Remove empty table cells pattern
        text = re.sub(r'\|\s*\|\s*\|\s*\|\s*\|', '', text)
        
        # Fix header formatting
        text = re.sub(r'^(#+\s*.*?)\n+', r'\1\n\n', text, flags=re.MULTILINE)
        
        # Improve paragraph spacing
        text = re.sub(r'([.!?])\s*\n([A-Z])', r'\1\n\n\2', text)
        
        # Remove redundant spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _add_image_references(self, results: Dict[str, Any], image_metadata: Dict[int, List[Dict]]) -> Dict[str, Any]:
        """Add image references to text content."""
        cross_references = []
        
        for page_info in results["pages"]:
            page_num = page_info["page_number"]
            if page_num in image_metadata:
                page_images = image_metadata[page_num]
                
                # Add image markers to text
                text_with_images = page_info["text"]
                
                for img in page_images:
                    img_filename = img.get("filename", "unknown")
                    img_type = img.get("classification", {}).get("type", "unknown")
                    coordinates = img.get("coordinates", {})
                    
                    # Create image reference
                    image_ref = f"\n\n[IMAGE: {img_filename} - {img_type} at ({coordinates.get('x', 0)}, {coordinates.get('y', 0)})]\n\n"
                    
                    # Insert reference at logical location (simplified approach)
                    # In a more sophisticated version, this would analyze text context
                    if "tip" in text_with_images.lower() and img_type in ["photo", "illustration"]:
                        # Insert image reference near tips
                        text_with_images = text_with_images.replace("Monthly savings tip:", f"Monthly savings tip:{image_ref}")
                    else:
                        # Add at end of page
                        text_with_images += image_ref
                    
                    # Create cross-reference entry
                    cross_references.append({
                        "page_number": page_num,
                        "image_filename": img_filename,
                        "image_type": img_type,
                        "coordinates": coordinates,
                        "text_context": self._extract_nearby_text(page_info["text"], coordinates),
                        "correlation_strength": self._calculate_correlation(page_info["text"], img)
                    })
                
                page_info["text"] = text_with_images
        
        # Update full text
        results["full_text"] = "\n\n".join([page["text"] for page in results["pages"]])
        results["cross_references"] = cross_references
        
        return results
    
    def _extract_nearby_text(self, text: str, coordinates: Dict[str, int]) -> str:
        """Extract text likely related to an image based on coordinates."""
        # Simplified approach - in production, this would use actual coordinate analysis
        lines = text.split('\n')
        
        # Look for contextual keywords
        context_lines = []
        for line in lines:
            if any(keyword in line.lower() for keyword in ["tip", "advice", "save", "energy", "use"]):
                context_lines.append(line.strip())
        
        return " ".join(context_lines[:3])  # Return first 3 relevant lines
    
    def _calculate_correlation(self, text: str, image_info: Dict[str, Any]) -> float:
        """Calculate correlation strength between text and image."""
        img_type = image_info.get("classification", {}).get("type", "")
        text_lower = text.lower()
        
        correlation = 0.0
        
        # Type-based correlations
        if img_type == "logo" and any(word in text_lower for word in ["company", "energy", "xcel"]):
            correlation += 0.3
        elif img_type == "photo" and any(word in text_lower for word in ["tip", "save", "laundry", "home"]):
            correlation += 0.4
        elif img_type == "chart" and any(word in text_lower for word in ["usage", "kwh", "typical"]):
            correlation += 0.5
        
        # OCR text correlation
        if image_info.get("text_detected", False):
            correlation += 0.2
        
        return min(correlation, 1.0)


def main():
    """Enhanced main function with comprehensive CLI."""
    parser = argparse.ArgumentParser(
        description="Enhanced PDF Text Extractor with Advanced Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_pdf_extractor.py energy_report.pdf --enhanced-layout
  python enhanced_pdf_extractor.py energy_report.pdf --with-images --cross-reference
  python enhanced_pdf_extractor.py energy_report.pdf --embedding-chunks --chunk-strategy semantic
  python enhanced_pdf_extractor.py energy_report.pdf --extract-structured-data
  python enhanced_pdf_extractor.py energy_report.pdf --output-formats markdown json
        """
    )
    
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--enhanced-layout", action="store_true", help="Enable enhanced layout analysis")
    parser.add_argument("--with-images", action="store_true", help="Include image cross-references")
    parser.add_argument("--cross-reference", action="store_true", help="Generate text-image cross-references")
    parser.add_argument("--embedding-chunks", action="store_true", help="Create embedding-optimized chunks")
    parser.add_argument("--chunk-strategy", choices=["semantic", "size", "structure", "hybrid"], 
                       default="semantic", help="Chunking strategy for embeddings")
    parser.add_argument("--chunk-size", type=int, default=512, help="Maximum chunk size for embeddings")
    parser.add_argument("--extract-structured-data", action="store_true", help="Extract structured data patterns")
    parser.add_argument("--output-formats", nargs="+", choices=["markdown", "json", "csv"], 
                       default=["markdown"], help="Output formats")
    parser.add_argument("--image-metadata", help="Path to image metadata JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-error output")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Initialize extractor
    extractor = EnhancedPDFTextExtractor(
        output_dir=args.output_dir,
        enable_structured_extraction=args.extract_structured_data or args.enhanced_layout
    )
    
    try:
        print(f"🔍 Processing PDF with enhanced extraction: {args.pdf_path}")
        
        # Determine extraction method
        if args.with_images or args.cross_reference:
            results = extractor.extract_with_image_references(args.pdf_path, args.image_metadata)
        else:
            results = extractor.extract_with_layout_analysis(args.pdf_path, args.enhanced_layout)
        
        # Create embedding chunks if requested
        if args.embedding_chunks:
            chunks = extractor.create_embedding_chunks(
                results["full_text"],
                args.chunk_strategy,
                results.get("cross_references"),
                args.chunk_size
            )
            results["embedding_chunks"] = chunks
        
        # Extract tables if advanced layout is enabled
        if args.enhanced_layout:
            tables = extractor.extract_tables_advanced(args.pdf_path)
            results["tables"] = tables
        
        # Save results
        pdf_name = pathlib.Path(args.pdf_path).stem
        saved_files = extractor.save_enhanced_results(results, pdf_name)
        
        # Print summary
        print(f"\n✅ Enhanced extraction completed successfully!")
        print(f"📄 Processed {results['total_pages']} pages")
        print(f"📝 Extracted {len(results['full_text'])} characters")
        
        if results.get("structured_data"):
            structured_count = sum(len(v) if isinstance(v, list) else 1 
                                 for v in results["structured_data"].values())
            print(f"🔍 Found {structured_count} structured data elements")
        
        if results.get("embedding_chunks"):
            print(f"📦 Created {len(results['embedding_chunks'])} embedding chunks")
        
        if results.get("cross_references"):
            print(f"🔗 Generated {len(results['cross_references'])} image cross-references")
        
        if results.get("tables"):
            print(f"📊 Extracted {len(results['tables'])} tables")
        
        print(f"\n📁 Output files:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {file_path}")
        
        print("\n🎉 Enhanced PDF processing completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()