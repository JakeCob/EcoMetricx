#!/usr/bin/env python3
"""
Test script for PDF Text Extractor

This script tests the functionality of the PDF extraction system
to ensure everything is working correctly.

Run with: python test_extraction.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add the current directory to path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pdf_extractor import PDFTextExtractor
    import pymupdf4llm
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure pymupdf4llm is installed: pip install pymupdf4llm")
    sys.exit(1)


class TestPDFExtractor(unittest.TestCase):
    """Test cases for the PDF Text Extractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_output_dir = tempfile.mkdtemp()
        self.extractor = PDFTextExtractor(self.test_output_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_output_dir, ignore_errors=True)
    
    def test_extractor_initialization(self):
        """Test that the extractor initializes correctly."""
        self.assertTrue(os.path.exists(self.test_output_dir))
        self.assertIsInstance(self.extractor, PDFTextExtractor)
    
    def test_document_stats(self):
        """Test document statistics calculation."""
        sample_text = "# Header\n\nSome text.\n\n| Col1 | Col2 |\n|------|------|\n| A    | B    |"
        stats = self.extractor.get_document_stats(sample_text)
        
        self.assertIn('total_characters', stats)
        self.assertIn('total_words', stats)
        self.assertIn('header_count', stats)
        self.assertIn('table_lines', stats)
        
        # Verify specific counts
        self.assertEqual(stats['header_count'], 1)
        self.assertEqual(stats['table_lines'], 3)  # Header, separator, data row
    
    def test_save_markdown(self):
        """Test markdown saving functionality."""
        test_content = "# Test Document\n\nThis is a test."
        output_path = self.extractor.save_markdown(test_content, "test.md")
        
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        
        self.assertEqual(saved_content, test_content)
    
    def test_save_page_chunks(self):
        """Test page chunks saving functionality."""
        test_chunks = [
            {"metadata": {"page": 0}, "text": "Page 1 content", "images": []},
            {"metadata": {"page": 1}, "text": "Page 2 content", "images": ["img1.png"]}
        ]
        
        output_path = self.extractor.save_page_chunks(test_chunks, "test_chunks.json")
        self.assertTrue(os.path.exists(output_path))
        
        # Verify JSON structure
        import json
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        self.assertIn('extraction_date', saved_data)
        self.assertIn('total_pages', saved_data)
        self.assertIn('pages', saved_data)
        self.assertEqual(saved_data['total_pages'], 2)


def test_pymupdf4llm_installation():
    """Test that pymupdf4llm is properly installed and functional."""
    print("\n🔍 Testing pymupdf4llm installation...")
    
    try:
        import pymupdf4llm
        print("✅ pymupdf4llm imported successfully")
        
        # Test available methods
        methods = [attr for attr in dir(pymupdf4llm) if not attr.startswith('_')]
        print(f"✅ Available methods: {', '.join(methods)}")
        
        # Check for key methods
        required_methods = ['to_markdown', 'LlamaMarkdownReader']
        for method in required_methods:
            if hasattr(pymupdf4llm, method):
                print(f"✅ {method} method available")
            else:
                print(f"❌ {method} method not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing pymupdf4llm: {e}")
        return False


def create_sample_pdf():
    """Create a simple sample PDF for testing if no PDF is provided."""
    try:
        import pymupdf  # PyMuPDF for creating test PDF
        
        doc = pymupdf.open()  # Create new PDF
        page = doc.new_page()
        
        # Add some text
        text = """Sample PDF Document
        
This is a test document for validating PDF extraction.

Key Information:
- Item 1: Sample data
- Item 2: More sample data

Table Example:
Category | Value
---------|-------
Test     | 123
Example  | 456
"""
        
        page.insert_text((50, 50), text)
        
        # Save to temp file
        temp_pdf = "sample_test.pdf"
        doc.save(temp_pdf)
        doc.close()
        
        print(f"✅ Created sample PDF: {temp_pdf}")
        return temp_pdf
        
    except ImportError:
        print("⚠️  PyMuPDF not available for creating sample PDF")
        return None
    except Exception as e:
        print(f"❌ Error creating sample PDF: {e}")
        return None


def run_integration_test():
    """Run integration test with actual PDF processing."""
    print("\n🧪 Running integration tests...")
    
    # Try to create a sample PDF
    sample_pdf = create_sample_pdf()
    
    if sample_pdf:
        try:
            extractor = PDFTextExtractor("test_output")
            
            # Test markdown extraction
            md_text = extractor.extract_text_to_markdown(sample_pdf)
            print(f"✅ Extracted {len(md_text)} characters of markdown text")
            
            # Test stats
            stats = extractor.get_document_stats(md_text)
            print(f"✅ Document stats: {stats['total_words']} words, {stats['header_count']} headers")
            
            # Save results
            md_file = extractor.save_markdown(md_text, "integration_test.md")
            print(f"✅ Saved markdown to: {md_file}")
            
            # Clean up
            os.remove(sample_pdf)
            print("✅ Integration test completed successfully")
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            if os.path.exists(sample_pdf):
                os.remove(sample_pdf)
    else:
        print("⚠️  Skipping integration test (no sample PDF created)")


def main():
    """Run all tests."""
    print("🚀 Starting PDF Extractor Tests")
    print("=" * 50)
    
    # Test installation
    if not test_pymupdf4llm_installation():
        print("\n❌ Installation test failed. Please check your setup.")
        return False
    
    # Run unit tests
    print("\n🧪 Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    run_integration_test()
    
    print("\n🎉 All tests completed!")
    print("\n📋 Quick Start Commands:")
    print("  python pdf_extractor.py task/test_info_extract.pdf")
    print("  python pdf_extractor.py task/test_info_extract.pdf --with-images")
    print("  python pdf_extractor.py task/test_info_extract.pdf --pages 0 1")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
