#!/usr/bin/env python3
"""
PDF Text Extractor using pymupdf4llm

This script extracts text and structural information from PDF documents
using the pymupdf4llm library, which converts PDF content to markdown format
optimized for LLM applications.

Author: Assistant
Date: 2025-09-02
"""

import os
import sys
import json
import pathlib
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

try:
    import pymupdf4llm
except ImportError:
    print("Error: pymupdf4llm is not installed.")
    print("Please install it using: pip install pymupdf4llm")
    sys.exit(1)


class PDFTextExtractor:
    """
    A class to extract text from PDF documents using pymupdf4llm.
    
    This class provides methods to extract text in various formats including
    markdown, page chunks, and with image extraction capabilities.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the PDF Text Extractor.
        
        Args:
            output_dir (str): Directory to save extracted content and images
        """
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_text_to_markdown(self, pdf_path: str, pages: Optional[List[int]] = None) -> str:
        """
        Extract text from PDF and convert to markdown format.
        
        Args:
            pdf_path (str): Path to the PDF file
            pages (Optional[List[int]]): List of 0-based page numbers to process.
                                       If None, all pages are processed.
        
        Returns:
            str: Extracted text in markdown format
        """
        try:
            self.logger.info(f"Extracting text from: {pdf_path}")
            
            if pages:
                self.logger.info(f"Processing specific pages: {pages}")
                md_text = pymupdf4llm.to_markdown(pdf_path, pages=pages)
            else:
                self.logger.info("Processing all pages")
                md_text = pymupdf4llm.to_markdown(pdf_path)
            
            self.logger.info(f"Successfully extracted {len(md_text)} characters")
            return md_text
            
        except Exception as e:
            self.logger.error(f"Error extracting text: {str(e)}")
            raise
    
    def extract_with_images(self, pdf_path: str, 
                          image_format: str = "png", 
                          dpi: int = 150) -> List[Dict[str, Any]]:
        """
        Extract text and images from PDF with page-by-page breakdown.
        
        Args:
            pdf_path (str): Path to the PDF file
            image_format (str): Format for extracted images (png, jpg, etc.)
            dpi (int): Resolution for extracted images
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing page data
        """
        try:
            self.logger.info(f"Extracting text and images from: {pdf_path}")
            
            image_path = self.output_dir / "images"
            image_path.mkdir(exist_ok=True)
            
            # Extract with images
            page_chunks = pymupdf4llm.to_markdown(
                pdf_path,
                page_chunks=True,
                write_images=True,
                image_path=str(image_path),
                image_format=image_format,
                dpi=dpi
            )
            
            self.logger.info(f"Extracted {len(page_chunks)} pages with images")
            return page_chunks
            
        except Exception as e:
            self.logger.error(f"Error extracting with images: {str(e)}")
            raise
    
    def save_markdown(self, content: str, filename: str = "extracted_text.md") -> str:
        """
        Save markdown content to a file.
        
        Args:
            content (str): Markdown content to save
            filename (str): Output filename
        
        Returns:
            str: Path to saved file
        """
        output_path = self.output_dir / filename
        
        try:
            output_path.write_text(content, encoding='utf-8')
            self.logger.info(f"Saved markdown to: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.error(f"Error saving markdown: {str(e)}")
            raise
    
    def save_page_chunks(self, page_chunks: List[Dict[str, Any]], 
                        filename: str = "page_chunks.json") -> str:
        """
        Save page chunks data to a JSON file.
        
        Args:
            page_chunks (List[Dict[str, Any]]): Page chunks data
            filename (str): Output filename
        
        Returns:
            str: Path to saved file
        """
        output_path = self.output_dir / filename
        
        try:
            # Convert to JSON-serializable format
            json_data = {
                "extraction_date": datetime.now().isoformat(),
                "total_pages": len(page_chunks),
                "pages": []
            }
            
            for i, chunk in enumerate(page_chunks):
                # Convert all chunk data to JSON-serializable format
                def make_serializable(obj):
                    """Recursively convert objects to JSON-serializable format."""
                    if isinstance(obj, dict):
                        return {key: make_serializable(value) for key, value in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [make_serializable(item) for item in obj]
                    elif isinstance(obj, (str, int, float, bool, type(None))):
                        return obj
                    else:
                        return str(obj)  # Convert non-serializable objects to strings
                
                page_data = {
                    "page_number": i,
                    "metadata": make_serializable(chunk.get("metadata", {})),
                    "text": chunk.get("text", ""),
                    "images": make_serializable(chunk.get("images", []))
                }
                json_data["pages"].append(page_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved page chunks to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving page chunks: {str(e)}")
            raise
    
    def get_document_stats(self, content: str) -> Dict[str, Any]:
        """
        Get basic statistics about the extracted content.
        
        Args:
            content (str): Extracted markdown content
        
        Returns:
            Dict[str, Any]: Statistics about the content
        """
        lines = content.split('\n')
        words = content.split()
        
        # Count headers
        headers = [line for line in lines if line.strip().startswith('#')]
        
        # Count tables (markdown tables start with |)
        table_lines = [line for line in lines if line.strip().startswith('|')]
        
        stats = {
            "total_characters": len(content),
            "total_words": len(words),
            "total_lines": len(lines),
            "header_count": len(headers),
            "table_lines": len(table_lines),
            "non_empty_lines": len([line for line in lines if line.strip()])
        }
        
        return stats


def main():
    """
    Main function to demonstrate PDF text extraction.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from PDF using pymupdf4llm")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--with-images", action="store_true", help="Extract images as well")
    parser.add_argument("--pages", nargs="+", type=int, help="Specific pages to extract (0-based)")
    parser.add_argument("--image-format", default="png", help="Image format (png, jpg, etc.)")
    parser.add_argument("--dpi", type=int, default=150, help="Image resolution DPI")
    
    args = parser.parse_args()
    
    # Validate PDF file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Initialize extractor
    extractor = PDFTextExtractor(args.output_dir)
    
    try:
        if args.with_images:
            # Extract with images
            page_chunks = extractor.extract_with_images(
                args.pdf_path, 
                args.image_format, 
                args.dpi
            )
            
            # Save page chunks
            chunks_file = extractor.save_page_chunks(page_chunks)
            print(f"✅ Saved page chunks with images to: {chunks_file}")
            
            # Also save combined markdown
            combined_text = "\n\n".join([chunk.get("text", "") for chunk in page_chunks])
            md_file = extractor.save_markdown(combined_text, "extracted_with_images.md")
            
        else:
            # Extract text only
            md_text = extractor.extract_text_to_markdown(args.pdf_path, args.pages)
            
            # Save markdown
            md_file = extractor.save_markdown(md_text)
            print(f"✅ Saved markdown to: {md_file}")
            
            # Show stats
            stats = extractor.get_document_stats(md_text)
            print("\n📊 Document Statistics:")
            for key, value in stats.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\n🎉 Text extraction completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during extraction: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
