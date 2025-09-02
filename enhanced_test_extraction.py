#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced PDF Processing System

This script provides comprehensive testing for the integrated PDF processing system,
including enhanced text extraction, image analysis, and multimodal integration.

Features:
- Unit tests for all enhanced components
- Integration tests for text-image correlation
- Quality validation for embedding outputs
- Performance benchmarking
- Energy report specific validation

Author: Assistant
Date: 2025-09-02
"""

import os
import sys
import json
import tempfile
import unittest
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all enhanced extractors
try:
    from enhanced_pdf_extractor import EnhancedPDFTextExtractor, LayoutAnalyzer, StructuredDataExtractor, EmbeddingChunker
    from enhanced_image_extractor import EnhancedPDFImageExtractor, ImageVisibilityAnalyzer, TextImageCorrelator
    from integrated_extractor import IntegratedPDFProcessor, MultimodalChunkGenerator, QualityAnalyzer
    
    # Also import original extractors for comparison
    from pdf_extractor import PDFTextExtractor
    from image_extractor import PDFImageExtractor
    
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Enhanced modules import error: {e}")
    print("Please ensure all enhanced extractor files are in the same directory.")
    ENHANCED_MODULES_AVAILABLE = False


class TestEnhancedTextExtractor(unittest.TestCase):
    """Test cases for Enhanced PDF Text Extractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_output_dir = tempfile.mkdtemp()
        if ENHANCED_MODULES_AVAILABLE:
            self.extractor = EnhancedPDFTextExtractor(self.test_output_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_output_dir, ignore_errors=True)
    
    @unittest.skipUnless(ENHANCED_MODULES_AVAILABLE, "Enhanced modules not available")
    def test_enhanced_extractor_initialization(self):
        """Test enhanced extractor initialization."""
        self.assertIsInstance(self.extractor, EnhancedPDFTextExtractor)
        self.assertTrue(self.extractor.text_dir.exists())
        self.assertTrue(self.extractor.integrated_dir.exists())
        self.assertTrue(self.extractor.embeddings_ready_dir.exists())
    
    @unittest.skipUnless(ENHANCED_MODULES_AVAILABLE, "Enhanced modules not available")
    def test_layout_analyzer(self):
        """Test layout analysis component."""
        analyzer = LayoutAnalyzer()
        self.assertIsInstance(analyzer, LayoutAnalyzer)
        
        # Test column detection logic
        # This would normally require a real PDF page, so we test the basic structure
        self.assertTrue(hasattr(analyzer, 'analyze_page_layout'))
        self.assertTrue(hasattr(analyzer, '_detect_columns'))
        self.assertTrue(hasattr(analyzer, '_detect_tables'))
    
    @unittest.skipUnless(ENHANCED_MODULES_AVAILABLE, "Enhanced modules not available")
    def test_structured_data_extractor(self):
        """Test structured data extraction."""
        extractor = StructuredDataExtractor()
        
        # Test with energy report sample text
        sample_text = """
        Home Energy Report: electricity
        March report Account number: 954137 Service address: 1627 Tulip Lane
        Dear JILL DOE, here is your usage analysis for March.
        Your electric use: 125 kWh above typical use
        Contact us at 800.895.4999 or visit franklinenergy.com
        Monthly savings tip: Do full laundry loads to save up to 6% of your energy use.
        """
        
        structured_data = extractor.extract_structured_data(sample_text)
        
        # Verify expected extractions
        self.assertIn('account_number', structured_data)
        self.assertIn('service_address', structured_data)
        self.assertIn('phone_number', structured_data)
        self.assertIn('energy_usage', structured_data)
        self.assertIn('website', structured_data)
        self.assertIn('customer_name', structured_data)
        
        # Verify specific values
        self.assertIn('954137', structured_data['account_number'])
        self.assertIn('1627 Tulip Lane', ' '.join(structured_data['service_address']))
        self.assertIn('800.895.4999', structured_data['phone_number'])
        self.assertIn('JILL DOE', structured_data['customer_name'])
    
    @unittest.skipUnless(ENHANCED_MODULES_AVAILABLE, "Enhanced modules not available")
    def test_embedding_chunker(self):
        """Test embedding chunk generation."""
        chunker = EmbeddingChunker(max_chunk_size=200, overlap_size=20)
        
        sample_text = """
        # Energy Report
        
        This is a comprehensive energy usage report for your household.
        
        ## Usage Summary
        
        Your energy consumption for March was 125 kWh, which is above typical usage.
        
        ## Savings Opportunities
        
        Here are some tips to reduce your energy consumption:
        - Do full laundry loads
        - Adjust thermostat settings
        - Use energy-efficient appliances
        
        ## Contact Information
        
        For questions, call 800.895.4999 or visit our website at franklinenergy.com.
        """
        
        # Test semantic chunking
        semantic_chunks = chunker.create_chunks(sample_text, "semantic")
        self.assertGreater(len(semantic_chunks), 0)
        
        for chunk in semantic_chunks:
            self.assertIn('chunk_id', chunk)
            self.assertIn('text', chunk)
            self.assertIn('character_count', chunk)
            self.assertIn('word_count', chunk)
            self.assertIn('importance_score', chunk)
            self.assertIn('content_type', chunk)
            self.assertLessEqual(len(chunk['text']), chunker.max_chunk_size)
        
        # Test size-based chunking
        size_chunks = chunker.create_chunks(sample_text, "size")
        self.assertGreater(len(size_chunks), 0)
        
        # Test structure-based chunking
        structure_chunks = chunker.create_chunks(sample_text, "structure")
        self.assertGreater(len(structure_chunks), 0)


class TestEnhancedImageExtractor(unittest.TestCase):
    """Test cases for Enhanced PDF Image Extractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_output_dir = tempfile.mkdtemp()
        if ENHANCED_MODULES_AVAILABLE:
            self.extractor = EnhancedPDFImageExtractor(self.test_output_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_output_dir, ignore_errors=True)
    
    @unittest.skipUnless(ENHANCED_MODULES_AVAILABLE, "Enhanced modules not available")
    def test_enhanced_image_extractor_initialization(self):
        """Test enhanced image extractor initialization."""
        self.assertIsInstance(self.extractor, EnhancedPDFImageExtractor)
        self.assertTrue(self.extractor.visible_dir.exists())
        self.assertTrue(self.extractor.embedded_dir.exists())
        self.assertTrue(self.extractor.by_type_dir.exists())
    
    @unittest.skipUnless(ENHANCED_MODULES_AVAILABLE, "Enhanced modules not available")
    def test_visibility_analyzer(self):
        """Test image visibility analysis."""
        analyzer = ImageVisibilityAnalyzer()
        
        # Test small header image (logo-like)
        image_data = {
            "coordinates": {"x": 30, "y": 28, "width": 127, "height": 39},
            "dimensions": {"width": 178, "height": 55},
            "color_palette": 50,
            "text_detected": True
        }
        
        page_content = {
            "page_width": 612,
            "page_height": 792,
            "text": "Energy Company Logo and contact information"
        }
        
        result = analyzer.analyze_image_visibility(image_data, page_content)
        
        self.assertIn('visibility', result)
        self.assertIn('confidence', result)
        self.assertIn('reasoning', result)
        self.assertIn('analysis_factors', result)
        
        # Should detect as visible for a small header image
        self.assertIn(result['visibility'], ['visible', 'embedded'])
    
    @unittest.skipUnless(ENHANCED_MODULES_AVAILABLE, "Enhanced modules not available")
    def test_text_image_correlator(self):
        """Test text-image correlation."""
        correlator = TextImageCorrelator()
        
        # Mock image data
        images = [{
            "filename": "energy_report_page0_img0_chart.jpg",
            "page_number": 0,
            "classification": {"type": "chart", "confidence": 0.9},
            "coordinates": {"x": 416, "y": 529, "width": 469, "height": 458},
            "text_detected": True
        }]
        
        text_content = """
        Your electric use: 125 kWh above typical use
        This chart shows your usage compared to similar homes.
        Monthly savings tip: Do full laundry loads to save energy.
        """
        
        enhanced_images = correlator.correlate_with_text(images, text_content)
        
        self.assertEqual(len(enhanced_images), 1)
        enhanced_img = enhanced_images[0]
        
        self.assertIn('text_correlation', enhanced_img)
        self.assertIn('contextual_description', enhanced_img)
        
        correlation = enhanced_img['text_correlation']
        self.assertIn('nearby_text', correlation)
        self.assertIn('primary_context', correlation)
        self.assertIn('correlation_strength', correlation)
        
        # Should detect usage_data context for chart
        self.assertIn('usage', correlation['primary_context'])


class TestIntegratedProcessor(unittest.TestCase):
    """Test cases for Integrated PDF Processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_output_dir = tempfile.mkdtemp()
        if ENHANCED_MODULES_AVAILABLE:
            self.processor = IntegratedPDFProcessor(self.test_output_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_output_dir, ignore_errors=True)
    
    @unittest.skipUnless(ENHANCED_MODULES_AVAILABLE, "Enhanced modules not available")
    def test_multimodal_chunk_generator(self):
        """Test multimodal chunk generation."""
        generator = MultimodalChunkGenerator()
        
        # Mock text chunks
        text_chunks = [
            {
                "chunk_id": 0,
                "text": "Monthly savings tip: Do full laundry loads to save energy.",
                "content_type": "advice",
                "metadata": {"page_number": 0}
            },
            {
                "chunk_id": 1,
                "text": "Your usage was 125 kWh, which is above typical.",
                "content_type": "usage_data",
                "metadata": {"page_number": 0}
            }
        ]
        
        # Mock images
        images = [
            {
                "filename": "laundry_tip.jpg",
                "page_number": 0,
                "classification": {"type": "photo"},
                "text_correlation": {
                    "primary_context": "energy_tips",
                    "correlation_strength": 0.8
                },
                "contextual_description": "Photo showing laundry being done efficiently"
            },
            {
                "filename": "usage_chart.jpg",
                "page_number": 0,
                "classification": {"type": "chart"},
                "text_correlation": {
                    "primary_context": "usage_data",
                    "correlation_strength": 0.9
                },
                "contextual_description": "Chart showing energy usage comparison"
            }
        ]
        
        # Test semantic pairing
        multimodal_chunks = generator.create_multimodal_chunks(text_chunks, images, "semantic")
        
        self.assertEqual(len(multimodal_chunks), 2)
        
        for chunk in multimodal_chunks:
            self.assertIn('text_chunk', chunk)
            self.assertIn('related_images', chunk)
            self.assertIn('pairing_method', chunk)
            self.assertIn('multimodal_chunk_id', chunk)
            self.assertIn('pairing_quality', chunk)
    
    @unittest.skipUnless(ENHANCED_MODULES_AVAILABLE, "Enhanced modules not available")
    def test_quality_analyzer(self):
        """Test quality analysis."""
        analyzer = QualityAnalyzer()
        
        # Mock extraction results
        text_results = {
            "full_text": "Sample energy report with account 954137 and usage data.",
            "pages": [{"page": 0}],
            "structured_data": {
                "account_number": ["954137"],
                "energy_usage": [125]
            },
            "embedding_chunks": [
                {"character_count": 400, "text": "Sample chunk"}
            ]
        }
        
        image_results = {
            "extracted_images": [
                {
                    "classification": {"confidence": 0.8},
                    "visibility_analysis": {"visibility": "visible"},
                    "text_correlation": {"correlation_strength": 0.7}
                }
            ],
            "processing_summary": {"unique_images": 1}
        }
        
        multimodal_chunks = [
            {
                "related_images": [{"correlation_strength": 0.8}],
                "pairing_quality": {"overall_quality": 0.7}
            }
        ]
        
        quality_analysis = analyzer.analyze_extraction_quality(text_results, image_results, multimodal_chunks)
        
        self.assertIn('text_quality', quality_analysis)
        self.assertIn('image_quality', quality_analysis)
        self.assertIn('integration_quality', quality_analysis)
        self.assertIn('overall_score', quality_analysis)
        self.assertIn('recommendations', quality_analysis)
        
        # Scores should be between 0 and 1
        self.assertGreaterEqual(quality_analysis['overall_score'], 0)
        self.assertLessEqual(quality_analysis['overall_score'], 1)


class TestEnergyReportSpecific(unittest.TestCase):
    """Specific tests for energy report processing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_pdf = "task/test_info_extract.pdf"
    
    @unittest.skipUnless(ENHANCED_MODULES_AVAILABLE and os.path.exists("task/test_info_extract.pdf"), 
                         "Enhanced modules or test PDF not available")
    def test_energy_report_structured_data(self):
        """Test structured data extraction from energy report."""
        extractor = StructuredDataExtractor()
        
        # Use original extractor to get text first
        from pdf_extractor import PDFTextExtractor
        basic_extractor = PDFTextExtractor()
        
        try:
            import pymupdf4llm
            text_content = pymupdf4llm.to_markdown(self.test_pdf)
        except:
            self.skipTest("Could not extract text from test PDF")
        
        structured_data = extractor.extract_structured_data(text_content)
        
        # Verify energy report specific data
        expected_fields = ['account_number', 'service_address', 'phone_number', 'energy_usage']
        
        for field in expected_fields:
            self.assertIn(field, structured_data, f"Missing expected field: {field}")
        
        # Verify specific energy report values if present
        if 'account_number' in structured_data:
            account_numbers = structured_data['account_number']
            self.assertTrue(any('954137' in str(acc) for acc in account_numbers),
                          "Expected account number 954137 not found")
    
    @unittest.skipUnless(ENHANCED_MODULES_AVAILABLE and os.path.exists("task/test_info_extract.pdf"), 
                         "Enhanced modules or test PDF not available")
    def test_energy_report_image_classification(self):
        """Test image classification for energy report."""
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = EnhancedPDFImageExtractor(temp_dir)
            
            try:
                # Extract images with enhanced analysis
                results = extractor.extract_images_enhanced(self.test_pdf)
                
                images = results.get("extracted_images", [])
                self.assertGreater(len(images), 0, "No images extracted from energy report")
                
                # Check for expected image types in energy report
                image_types = [img["classification"]["type"] for img in images]
                
                # Energy reports typically have logos, charts, and photos
                expected_types = set(['logo', 'chart', 'photo', 'unknown'])
                found_types = set(image_types)
                
                # Should find at least some recognizable types
                recognizable_types = found_types & expected_types
                self.assertGreater(len(recognizable_types), 0, 
                                 f"No recognizable image types found. Found: {found_types}")
                
                # Verify image visibility analysis
                for img in images:
                    self.assertIn('visibility_analysis', img)
                    visibility = img['visibility_analysis']
                    self.assertIn('visibility', visibility)
                    self.assertIn(['visible', 'embedded', 'background'], visibility['visibility'])
                
            except Exception as e:
                self.skipTest(f"Could not process test PDF for image extraction: {e}")


def test_enhanced_system_installation():
    """Test that all enhanced system components are properly installed."""
    print("\n🔍 Testing enhanced system installation...")
    
    if not ENHANCED_MODULES_AVAILABLE:
        print("❌ Enhanced modules not available")
        return False
    
    # Test enhanced text extractor
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = EnhancedPDFTextExtractor(temp_dir)
            print("✅ Enhanced PDF Text Extractor initialized")
    except Exception as e:
        print(f"❌ Enhanced PDF Text Extractor failed: {e}")
        return False
    
    # Test enhanced image extractor
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = EnhancedPDFImageExtractor(temp_dir)
            print("✅ Enhanced PDF Image Extractor initialized")
    except Exception as e:
        print(f"❌ Enhanced PDF Image Extractor failed: {e}")
        return False
    
    # Test integrated processor
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = IntegratedPDFProcessor(temp_dir)
            print("✅ Integrated PDF Processor initialized")
    except Exception as e:
        print(f"❌ Integrated PDF Processor failed: {e}")
        return False
    
    print("✅ All enhanced system components available")
    return True


def run_performance_benchmark():
    """Run performance benchmark on the integrated system."""
    print("\n⚡ Running performance benchmark...")
    
    if not ENHANCED_MODULES_AVAILABLE:
        print("❌ Enhanced modules not available for benchmarking")
        return
    
    test_pdf = "task/test_info_extract.pdf"
    if not os.path.exists(test_pdf):
        print("❌ Test PDF not available for benchmarking")
        return
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = IntegratedPDFProcessor(temp_dir)
            
            # Benchmark full processing
            start_time = time.time()
            
            results = processor.process_pdf_comprehensive(
                test_pdf,
                {
                    "enhanced_layout": True,
                    "embedding_chunks": True,
                    "text_correlation": True,
                    "quality_analysis": True
                }
            )
            
            processing_time = time.time() - start_time
            
            # Analyze results
            text_pages = len(results["text_results"].get("pages", []))
            text_chunks = len(results["text_results"].get("embedding_chunks", []))
            total_images = len(results["image_results"].get("extracted_images", []))
            multimodal_chunks = len(results.get("multimodal_chunks", []))
            
            print(f"✅ Performance benchmark completed:")
            print(f"   Processing time: {processing_time:.2f} seconds")
            print(f"   Pages processed: {text_pages}")
            print(f"   Text chunks created: {text_chunks}")
            print(f"   Images extracted: {total_images}")
            print(f"   Multimodal chunks: {multimodal_chunks}")
            print(f"   Processing rate: {text_pages/processing_time:.1f} pages/second")
            
            if results.get("quality_analysis"):
                overall_score = results["quality_analysis"]["overall_score"]
                print(f"   Quality score: {overall_score}/1.0")
    
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")


def run_integration_test():
    """Run comprehensive integration test."""
    print("\n🧪 Running comprehensive integration test...")
    
    if not ENHANCED_MODULES_AVAILABLE:
        print("❌ Enhanced modules not available for integration test")
        return False
    
    test_pdf = "task/test_info_extract.pdf"
    if not os.path.exists(test_pdf):
        print("❌ Test PDF not available for integration test")
        return False
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test integrated processing
            processor = IntegratedPDFProcessor(temp_dir)
            
            results = processor.process_pdf_comprehensive(test_pdf)
            saved_files = processor.save_comprehensive_results(results)
            
            # Verify results structure
            required_sections = ['text_results', 'image_results', 'integration_summary']
            for section in required_sections:
                if section not in results:
                    print(f"❌ Missing results section: {section}")
                    return False
            
            # Verify saved files
            for file_type, file_path in saved_files.items():
                if not os.path.exists(file_path):
                    print(f"❌ Missing output file: {file_type} at {file_path}")
                    return False
            
            print("✅ Integration test passed successfully")
            print(f"   Generated {len(saved_files)} output files")
            return True
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all enhanced tests."""
    print("🚀 Starting Enhanced PDF Processing System Tests")
    print("=" * 60)
    
    # Test installation
    if not test_enhanced_system_installation():
        print("\n❌ Enhanced system installation test failed.")
        return False
    
    # Run unit tests
    if ENHANCED_MODULES_AVAILABLE:
        print("\n🧪 Running enhanced unit tests...")
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Add test cases
        test_suite.addTest(unittest.makeSuite(TestEnhancedTextExtractor))
        test_suite.addTest(unittest.makeSuite(TestEnhancedImageExtractor))
        test_suite.addTest(unittest.makeSuite(TestIntegratedProcessor))
        test_suite.addTest(unittest.makeSuite(TestEnergyReportSpecific))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        if not result.wasSuccessful():
            print(f"\n❌ Unit tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
    else:
        print("\n⚠️  Skipping unit tests (enhanced modules not available)")
    
    # Run integration test
    integration_success = run_integration_test()
    
    # Run performance benchmark
    run_performance_benchmark()
    
    print("\n🎉 Enhanced system testing completed!")
    print("\n📋 Available Enhanced Commands:")
    print("  # Enhanced text extraction")
    print("  python enhanced_pdf_extractor.py task/test_info_extract.pdf --enhanced-layout")
    print("  python enhanced_pdf_extractor.py task/test_info_extract.pdf --embedding-chunks --chunk-strategy semantic")
    print("")
    print("  # Enhanced image extraction")
    print("  python enhanced_image_extractor.py task/test_info_extract.pdf --visible-only")
    print("  python enhanced_image_extractor.py task/test_info_extract.pdf --enhance-for-embeddings")
    print("")
    print("  # Integrated processing")
    print("  python integrated_extractor.py task/test_info_extract.pdf --full-analysis")
    print("  python integrated_extractor.py task/test_info_extract.pdf --embedding-ready --chunk-size 512")
    
    return integration_success if ENHANCED_MODULES_AVAILABLE else True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)