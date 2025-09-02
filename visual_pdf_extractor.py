#!/usr/bin/env python3
"""
Visual PDF Extraction System for EcoMetricx
==========================================

A comprehensive PDF text extraction system using visual rendering + OCR to solve
text extraction issues where programmatic methods get "invisible data" instead 
of what's visually displayed.

This system provides:
- Screenshot-based extraction with high-quality OCR
- PDFPlumber alternative for complex layouts
- Hybrid extraction with intelligent fallback
- Comprehensive diagnostic system
- Image preprocessing for optimal OCR results
- Specialized handling for energy reports

Author: Claude (Anthropic)
Version: 1.0.0
"""

import os
import sys
import json
import logging
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Core dependencies
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Optional OpenCV for advanced preprocessing
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

# PDF processing
from pdf2image import convert_from_path
import pdfplumber
import pytesseract

# Standard PDF libraries for comparison
import fitz  # PyMuPDF
try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False

import re
from multiprocessing import Pool, cpu_count
import concurrent.futures


class OCRConfig:
    """Optimized OCR settings for different content types"""
    
    DOCUMENT_MODE = '--oem 3 --psm 6'
    TABLE_MODE = '--oem 3 --psm 6'
    MIXED_MODE = '--oem 3 --psm 3'
    SINGLE_LINE = '--oem 3 --psm 7'
    HIGH_ACCURACY = '--oem 3 --psm 6 -c tessedit_do_invert=0'
    
    @classmethod
    def get_config_for_content(cls, content_type: str) -> str:
        """Return optimal OCR config based on detected content"""
        config_map = {
            "document": cls.DOCUMENT_MODE,
            "table": cls.TABLE_MODE,
            "mixed": cls.MIXED_MODE,
            "single_line": cls.SINGLE_LINE,
            "high_accuracy": cls.HIGH_ACCURACY
        }
        return config_map.get(content_type, cls.DOCUMENT_MODE)


class ImagePreprocessor:
    """Advanced image preprocessing pipeline for optimal OCR results"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ImagePreprocessor")
    
    def preprocess_for_ocr(self, image: Image.Image, enhancement_level: str = "standard") -> Image.Image:
        """Enhance image quality for better OCR results"""
        try:
            if OPENCV_AVAILABLE:
                # Use OpenCV for advanced preprocessing
                return self._opencv_preprocessing(image, enhancement_level)
            else:
                # Fallback to PIL-based preprocessing
                return self._pil_preprocessing(image, enhancement_level)
            
        except Exception as e:
            self.logger.error(f"Error in image preprocessing: {str(e)}")
            return image
    
    def _opencv_preprocessing(self, image: Image.Image, enhancement_level: str) -> Image.Image:
        """OpenCV-based preprocessing (if available)"""
        # Convert to numpy array for OpenCV operations
        img_array = np.array(image)
        
        # Convert to grayscale if not already
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        if enhancement_level == "aggressive":
            enhanced = self._aggressive_preprocessing(gray)
        elif enhancement_level == "minimal":
            enhanced = self._minimal_preprocessing(gray)
        else:
            enhanced = self._standard_preprocessing(gray)
        
        return Image.fromarray(enhanced)
    
    def _pil_preprocessing(self, image: Image.Image, enhancement_level: str) -> Image.Image:
        """PIL-based preprocessing fallback"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        if enhancement_level == "aggressive":
            # High contrast and sharpening
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
        elif enhancement_level == "minimal":
            # Light sharpening only
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
        else:
            # Standard enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
        
        return image
    
    def _standard_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Standard image enhancement pipeline (requires OpenCV)"""
        if not OPENCV_AVAILABLE:
            return gray
            
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoising(enhanced)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def _aggressive_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Aggressive enhancement for poor quality images (requires OpenCV)"""
        if not OPENCV_AVAILABLE:
            return gray
            
        # Strong histogram equalization
        enhanced = cv2.equalizeHist(gray)
        
        # More aggressive denoising
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Morphological operations to clean up text
        kernel = np.ones((2,2), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Strong sharpening
        kernel = np.array([[-1,-1,-1,-1,-1],
                          [-1,2,2,2,-1],
                          [-1,2,8,2,-1],
                          [-1,2,2,2,-1],
                          [-1,-1,-1,-1,-1]])/8.0
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def _minimal_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Minimal processing to avoid artifacts (requires OpenCV)"""
        if not OPENCV_AVAILABLE:
            return gray
            
        # Light denoising only
        enhanced = cv2.fastNlMeansDenoising(gray, h=10)
        return enhanced
    
    def adaptive_preprocessing(self, image: Image.Image, initial_confidence: float) -> Image.Image:
        """Apply preprocessing based on initial OCR quality"""
        if initial_confidence < 50:
            self.logger.info("Low confidence detected, applying aggressive preprocessing")
            return self.preprocess_for_ocr(image, "aggressive")
        elif initial_confidence > 80:
            self.logger.info("High confidence detected, applying minimal preprocessing")
            return self.preprocess_for_ocr(image, "minimal")
        else:
            self.logger.info("Moderate confidence detected, applying standard preprocessing")
            return self.preprocess_for_ocr(image, "standard")


class VisualPDFExtractor:
    """Extract text using visual rendering + OCR - most reliable method"""
    
    def __init__(self, dpi: int = 300, output_dir: str = "output/visual_extraction"):
        self.dpi = dpi
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "screenshots").mkdir(exist_ok=True)
        (self.output_dir / "preprocessed").mkdir(exist_ok=True)
        (self.output_dir / "ocr_results").mkdir(exist_ok=True)
        (self.output_dir / "final_text").mkdir(exist_ok=True)
        
        self.preprocessor = ImagePreprocessor()
        self.logger = logging.getLogger(f"{__name__}.VisualPDFExtractor")
        
        # OCR configuration
        self.ocr_config = OCRConfig.DOCUMENT_MODE
        self.preprocessing_enabled = True
    
    def extract_via_screenshot(self, pdf_path: str, preprocess: bool = True) -> Dict[str, Any]:
        """Convert PDF pages to images, then OCR"""
        try:
            pdf_path = Path(pdf_path)
            base_name = pdf_path.stem
            
            self.logger.info(f"Starting visual extraction of {pdf_path}")
            
            # Step 1: Convert PDF pages to high-DPI images
            self.logger.info(f"Converting PDF to images at {self.dpi} DPI")
            images = convert_from_path(str(pdf_path), dpi=self.dpi)
            
            # Step 2: Process each page
            extraction_results = {
                "source_pdf": str(pdf_path),
                "extraction_method": "visual_ocr",
                "extraction_date": datetime.now().isoformat(),
                "total_pages": len(images),
                "dpi": self.dpi,
                "preprocessing_enabled": preprocess,
                "pages": [],
                "full_text": "",
                "confidence_scores": [],
                "processing_stats": {}
            }
            
            page_texts = []
            total_confidence = 0
            
            for page_num, image in enumerate(images):
                self.logger.info(f"Processing page {page_num + 1}/{len(images)}")
                
                # Save original screenshot
                screenshot_path = self.output_dir / "screenshots" / f"{base_name}_page{page_num}.png"
                image.save(screenshot_path, "PNG", quality=100, dpi=(self.dpi, self.dpi))
                
                # Preprocess if enabled
                if preprocess:
                    processed_image = self.preprocessor.preprocess_for_ocr(image)
                    preprocessed_path = self.output_dir / "preprocessed" / f"{base_name}_page{page_num}_enhanced.png"
                    processed_image.save(preprocessed_path)
                else:
                    processed_image = image
                
                # Perform OCR
                page_result = self._ocr_single_page(processed_image, page_num, base_name)
                extraction_results["pages"].append(page_result)
                
                page_texts.append(page_result["text"])
                total_confidence += page_result["confidence"]
            
            # Combine all text
            full_text = "\n\n".join(page_texts)
            extraction_results["full_text"] = full_text
            extraction_results["average_confidence"] = total_confidence / len(images) if images else 0
            
            # Save final text
            final_text_path = self.output_dir / "final_text" / f"{base_name}_visual.md"
            with open(final_text_path, 'w', encoding='utf-8') as f:
                f.write(f"# Visual OCR Extraction: {base_name}\n\n")
                f.write(f"**Extraction Date:** {extraction_results['extraction_date']}\n")
                f.write(f"**Average Confidence:** {extraction_results['average_confidence']:.1f}%\n")
                f.write(f"**Total Pages:** {extraction_results['total_pages']}\n\n")
                f.write("---\n\n")
                f.write(full_text)
            
            # Extract structured data for energy reports
            if self._is_energy_report(full_text):
                extraction_results["structured_data"] = self._extract_energy_report_data(full_text)
            
            self.logger.info(f"Visual extraction completed. Average confidence: {extraction_results['average_confidence']:.1f}%")
            return extraction_results
            
        except Exception as e:
            self.logger.error(f"Error in visual extraction: {str(e)}")
            raise
    
    def _ocr_single_page(self, image: Image.Image, page_num: int, base_name: str) -> Dict[str, Any]:
        """Perform OCR on a single page image"""
        try:
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(image, config=self.ocr_config, output_type=pytesseract.Output.DICT)
            
            # Extract text with confidence
            text_with_confidence = []
            confidences = []
            
            for i, word in enumerate(ocr_data['text']):
                if word.strip():
                    conf = int(ocr_data['conf'][i])
                    if conf > 0:  # Filter out very low confidence detections
                        text_with_confidence.append(word)
                        confidences.append(conf)
            
            # Combine text
            extracted_text = ' '.join(text_with_confidence)
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Save OCR results
            ocr_result = {
                "page_number": page_num,
                "text": extracted_text,
                "confidence": avg_confidence,
                "word_count": len(text_with_confidence),
                "raw_ocr_data": {
                    "words": len(text_with_confidence),
                    "confidence_distribution": {
                        "min": min(confidences) if confidences else 0,
                        "max": max(confidences) if confidences else 0,
                        "avg": avg_confidence,
                        "std": np.std(confidences) if confidences else 0
                    }
                }
            }
            
            # Save detailed OCR data
            ocr_results_path = self.output_dir / "ocr_results" / f"{base_name}_page{page_num}_ocr.json"
            with open(ocr_results_path, 'w', encoding='utf-8') as f:
                json.dump(ocr_result, f, indent=2, ensure_ascii=False)
            
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"Error in OCR for page {page_num}: {str(e)}")
            return {
                "page_number": page_num,
                "text": "",
                "confidence": 0,
                "word_count": 0,
                "error": str(e)
            }
    
    def _is_energy_report(self, text: str) -> bool:
        """Detect if this is an energy/utility report"""
        energy_indicators = [
            r'\bkWh\b', r'\benergy\b', r'\busage\b', r'\belectric\b',
            r'\butility\b', r'\bbill\b', r'\baccount\s+#?\d+',
            r'\bservice\s+address\b', r'\benergy\s+tip\b'
        ]
        
        matches = 0
        for pattern in energy_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        
        return matches >= 3  # Need at least 3 indicators
    
    def _extract_energy_report_data(self, text: str) -> Dict[str, List[str]]:
        """Extract structured data from energy reports"""
        patterns = {
            "account_number": r'\b(?:account\s*#?\s*|acct\s*#?\s*)?(\d{6,8})\b',
            "service_address": r'\b(\d{3,5}\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Lane|Ln|Drive|Dr|Road|Rd|Court|Ct|Circle|Cir|Boulevard|Blvd|Way))\b',
            "customer_name": r'\b([A-Z]{2,}\s+[A-Z]{2,}(?:\s+[A-Z]{2,})?)\b',
            "phone_number": r'\b(\d{3}[-.]?\d{3}[-.]?\d{4})\b',
            "website": r'\b([a-zA-Z0-9.-]+\.(?:com|net|org|gov|edu))\b',
            "energy_usage": r'\b(\d{1,4})\s*kWh\b',
            "percentage": r'\b(\d{1,3})%\b',
            "dollar_amount": r'\$\s?(\d{1,4}(?:\.\d{2})?)\b',
            "date": r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b'
        }
        
        structured_data = {}
        
        for field, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            # Remove duplicates while preserving order
            unique_matches = list(dict.fromkeys(matches))
            structured_data[field] = unique_matches
        
        return structured_data


class AlternativePDFExtractor:
    """Use PDFPlumber as alternative - often better than PyMuPDF for visible text"""
    
    def __init__(self, output_dir: str = "output/visual_extraction"):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(f"{__name__}.AlternativePDFExtractor")
    
    def extract_via_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Extract using PDFPlumber's visual-focused approach"""
        try:
            pdf_path = Path(pdf_path)
            base_name = pdf_path.stem
            
            self.logger.info(f"Starting PDFPlumber extraction of {pdf_path}")
            
            extraction_results = {
                "source_pdf": str(pdf_path),
                "extraction_method": "pdfplumber",
                "extraction_date": datetime.now().isoformat(),
                "pages": [],
                "full_text": "",
                "tables": [],
                "metadata": {}
            }
            
            page_texts = []
            all_tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                extraction_results["total_pages"] = len(pdf.pages)
                extraction_results["metadata"] = pdf.metadata or {}
                
                for page_num, page in enumerate(pdf.pages):
                    self.logger.debug(f"Processing page {page_num + 1}/{len(pdf.pages)}")
                    
                    # Extract text
                    page_text = page.extract_text() or ""
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    
                    page_info = {
                        "page_number": page_num,
                        "text": page_text,
                        "tables": page_tables,
                        "char_count": len(page_text),
                        "table_count": len(page_tables) if page_tables else 0
                    }
                    
                    extraction_results["pages"].append(page_info)
                    page_texts.append(page_text)
                    
                    if page_tables:
                        all_tables.extend(page_tables)
            
            # Combine all text
            full_text = "\n\n".join(page_texts)
            extraction_results["full_text"] = full_text
            extraction_results["tables"] = all_tables
            
            # Save results
            final_text_path = self.output_dir / "final_text" / f"{base_name}_pdfplumber.md"
            with open(final_text_path, 'w', encoding='utf-8') as f:
                f.write(f"# PDFPlumber Extraction: {base_name}\n\n")
                f.write(f"**Extraction Date:** {extraction_results['extraction_date']}\n")
                f.write(f"**Total Pages:** {extraction_results['total_pages']}\n")
                f.write(f"**Tables Found:** {len(all_tables)}\n\n")
                f.write("---\n\n")
                f.write(full_text)
                
                if all_tables:
                    f.write("\n\n## Extracted Tables\n\n")
                    for i, table in enumerate(all_tables):
                        f.write(f"### Table {i + 1}\n\n")
                        if table:
                            # Convert table to markdown
                            for row in table:
                                if row:
                                    f.write("| " + " | ".join(str(cell) if cell else "" for cell in row) + " |\n")
                        f.write("\n")
            
            self.logger.info(f"PDFPlumber extraction completed. {len(page_texts)} pages, {len(all_tables)} tables")
            return extraction_results
            
        except Exception as e:
            self.logger.error(f"Error in PDFPlumber extraction: {str(e)}")
            raise


class PDFExtractionDiagnostic:
    """Compare different extraction methods to identify best approach"""
    
    def __init__(self, output_dir: str = "output/diagnostic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.PDFExtractionDiagnostic")
    
    def diagnose_pdf_structure(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze which extraction method works best"""
        try:
            pdf_path = Path(pdf_path)
            base_name = pdf_path.stem
            
            self.logger.info(f"Starting diagnostic analysis of {pdf_path}")
            
            diagnostic_results = {
                "source_pdf": str(pdf_path),
                "analysis_date": datetime.now().isoformat(),
                "methods_tested": [],
                "recommendations": {},
                "method_results": {}
            }
            
            # Test Method 1: PyMuPDF (current standard)
            try:
                self.logger.info("Testing PyMuPDF extraction...")
                pymupdf_result = self._test_pymupdf(pdf_path)
                diagnostic_results["method_results"]["pymupdf"] = pymupdf_result
                diagnostic_results["methods_tested"].append("pymupdf")
            except Exception as e:
                self.logger.warning(f"PyMuPDF extraction failed: {str(e)}")
                diagnostic_results["method_results"]["pymupdf"] = {"error": str(e)}
            
            # Test Method 2: PyMuPDF4LLM (if available)
            if PYMUPDF4LLM_AVAILABLE:
                try:
                    self.logger.info("Testing PyMuPDF4LLM extraction...")
                    pymupdf4llm_result = self._test_pymupdf4llm(pdf_path)
                    diagnostic_results["method_results"]["pymupdf4llm"] = pymupdf4llm_result
                    diagnostic_results["methods_tested"].append("pymupdf4llm")
                except Exception as e:
                    self.logger.warning(f"PyMuPDF4LLM extraction failed: {str(e)}")
                    diagnostic_results["method_results"]["pymupdf4llm"] = {"error": str(e)}
            
            # Test Method 3: PDFPlumber
            try:
                self.logger.info("Testing PDFPlumber extraction...")
                pdfplumber_result = self._test_pdfplumber(pdf_path)
                diagnostic_results["method_results"]["pdfplumber"] = pdfplumber_result
                diagnostic_results["methods_tested"].append("pdfplumber")
            except Exception as e:
                self.logger.warning(f"PDFPlumber extraction failed: {str(e)}")
                diagnostic_results["method_results"]["pdfplumber"] = {"error": str(e)}
            
            # Test Method 4: Visual OCR (sample only - resource intensive)
            try:
                self.logger.info("Testing Visual OCR extraction (sample page)...")
                visual_result = self._test_visual_sample(pdf_path)
                diagnostic_results["method_results"]["visual_ocr"] = visual_result
                diagnostic_results["methods_tested"].append("visual_ocr")
            except Exception as e:
                self.logger.warning(f"Visual OCR extraction failed: {str(e)}")
                diagnostic_results["method_results"]["visual_ocr"] = {"error": str(e)}
            
            # Analyze results and make recommendations
            diagnostic_results["recommendations"] = self._analyze_methods(diagnostic_results["method_results"])
            
            # Save diagnostic results
            diagnostic_path = self.output_dir / f"{base_name}_diagnostic_report.json"
            with open(diagnostic_path, 'w', encoding='utf-8') as f:
                json.dump(diagnostic_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Diagnostic analysis completed. Recommended method: {diagnostic_results['recommendations'].get('best_method', 'unknown')}")
            return diagnostic_results
            
        except Exception as e:
            self.logger.error(f"Error in diagnostic analysis: {str(e)}")
            raise
    
    def _test_pymupdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Test PyMuPDF extraction"""
        doc = fitz.open(str(pdf_path))
        
        text_parts = []
        total_chars = 0
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            text_parts.append(text)
            total_chars += len(text)
        
        doc.close()
        
        full_text = "\n\n".join(text_parts)
        
        return {
            "method": "pymupdf",
            "total_pages": len(text_parts),
            "total_characters": total_chars,
            "avg_chars_per_page": total_chars / len(text_parts) if text_parts else 0,
            "text_sample": full_text[:500] + "..." if len(full_text) > 500 else full_text,
            "quality_indicators": self._assess_text_quality(full_text)
        }
    
    def _test_pymupdf4llm(self, pdf_path: Path) -> Dict[str, Any]:
        """Test PyMuPDF4LLM extraction"""
        text = pymupdf4llm.to_markdown(str(pdf_path))
        
        return {
            "method": "pymupdf4llm",
            "total_characters": len(text),
            "text_sample": text[:500] + "..." if len(text) > 500 else text,
            "quality_indicators": self._assess_text_quality(text)
        }
    
    def _test_pdfplumber(self, pdf_path: Path) -> Dict[str, Any]:
        """Test PDFPlumber extraction"""
        text_parts = []
        total_chars = 0
        table_count = 0
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
                total_chars += len(page_text)
                
                page_tables = page.extract_tables()
                if page_tables:
                    table_count += len(page_tables)
        
        full_text = "\n\n".join(text_parts)
        
        return {
            "method": "pdfplumber",
            "total_pages": len(text_parts),
            "total_characters": total_chars,
            "avg_chars_per_page": total_chars / len(text_parts) if text_parts else 0,
            "tables_found": table_count,
            "text_sample": full_text[:500] + "..." if len(full_text) > 500 else full_text,
            "quality_indicators": self._assess_text_quality(full_text)
        }
    
    def _test_visual_sample(self, pdf_path: Path) -> Dict[str, Any]:
        """Test Visual OCR on first page only (for diagnostic)"""
        # Convert only first page
        images = convert_from_path(str(pdf_path), dpi=200, first_page=1, last_page=1)
        
        if not images:
            return {"error": "No images generated"}
        
        image = images[0]
        
        # Simple OCR without heavy preprocessing
        text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
        confidence_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Calculate average confidence
        confidences = [int(conf) for conf in confidence_data['conf'] if int(conf) > 0]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            "method": "visual_ocr_sample",
            "pages_tested": 1,
            "total_characters": len(text),
            "average_confidence": avg_confidence,
            "text_sample": text[:500] + "..." if len(text) > 500 else text,
            "quality_indicators": self._assess_text_quality(text)
        }
    
    def _assess_text_quality(self, text: str) -> Dict[str, Any]:
        """Assess the quality of extracted text"""
        if not text:
            return {"score": 0, "issues": ["no_text_extracted"]}
        
        issues = []
        score = 100
        
        # Check for common extraction issues
        
        # 1. Too many repeated characters (indicates OCR issues)
        repeated_chars = len(re.findall(r'(.)\1{3,}', text))
        if repeated_chars > 5:
            issues.append("excessive_repeated_characters")
            score -= 20
        
        # 2. Too many non-ASCII characters (might indicate encoding issues)
        non_ascii = sum(1 for char in text if ord(char) > 127)
        if non_ascii / len(text) > 0.1:
            issues.append("high_non_ascii_content")
            score -= 15
        
        # 3. Very short text (likely incomplete extraction)
        if len(text.strip()) < 100:
            issues.append("very_short_text")
            score -= 30
        
        # 4. No proper sentences (no periods, question marks, etc.)
        sentence_endings = len(re.findall(r'[.!?]', text))
        if sentence_endings == 0 and len(text) > 50:
            issues.append("no_sentence_structure")
            score -= 25
        
        # 5. Excessive whitespace issues
        excessive_spaces = len(re.findall(r'\s{5,}', text))
        if excessive_spaces > 10:
            issues.append("excessive_whitespace")
            score -= 10
        
        # 6. Common extraction artifacts
        artifacts = ['Col1', 'Col2', 'Col3', '��', 'NULL', 'undefined']
        for artifact in artifacts:
            if artifact in text:
                issues.append(f"contains_artifact_{artifact}")
                score -= 5
        
        return {
            "score": max(0, score),
            "character_count": len(text),
            "word_count": len(text.split()),
            "sentence_count": sentence_endings,
            "issues": issues,
            "text_density": len(text.strip().replace(' ', '')) / len(text) if text else 0
        }
    
    def _analyze_methods(self, method_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze method results and recommend best approach"""
        method_scores = {}
        
        for method, result in method_results.items():
            if "error" in result:
                method_scores[method] = 0
                continue
            
            quality_score = result.get("quality_indicators", {}).get("score", 0)
            
            # Bonus points for specific capabilities
            if method == "pdfplumber" and result.get("tables_found", 0) > 0:
                quality_score += 10  # PDFPlumber is good with tables
            
            if method == "visual_ocr" and result.get("average_confidence", 0) > 70:
                quality_score += 15  # High-confidence OCR is very reliable
            
            method_scores[method] = quality_score
        
        # Find best method
        best_method = max(method_scores, key=method_scores.get) if method_scores else None
        best_score = method_scores.get(best_method, 0) if best_method else 0
        
        recommendations = {
            "best_method": best_method,
            "best_score": best_score,
            "method_scores": method_scores,
            "reasoning": self._get_recommendation_reasoning(best_method, best_score, method_results)
        }
        
        return recommendations
    
    def _get_recommendation_reasoning(self, best_method: str, best_score: float, method_results: Dict[str, Any]) -> str:
        """Provide reasoning for the recommendation"""
        if not best_method:
            return "No extraction method succeeded"
        
        if best_score < 30:
            return f"{best_method} scored highest ({best_score}) but quality is still poor. Consider visual OCR with preprocessing."
        elif best_score < 60:
            return f"{best_method} scored highest ({best_score}) with moderate quality. May need fallback methods."
        elif best_method == "visual_ocr":
            return f"Visual OCR recommended ({best_score}) - programmatic methods likely have invisible text issues"
        elif best_method == "pdfplumber":
            return f"PDFPlumber recommended ({best_score}) - good for complex layouts and tables"
        else:
            return f"{best_method} recommended ({best_score}) - standard extraction works well"


class HybridPDFExtractor:
    """Intelligent fallback between extraction methods"""
    
    def __init__(self, output_dir: str = "output/combined"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.visual_extractor = VisualPDFExtractor()
        self.alternative_extractor = AlternativePDFExtractor()
        self.diagnostic = PDFExtractionDiagnostic()
        
        self.logger = logging.getLogger(f"{__name__}.HybridPDFExtractor")
    
    def extract_with_fallback(self, pdf_path: str, auto_fallback: bool = True) -> Dict[str, Any]:
        """Try methods in order of speed vs reliability"""
        try:
            pdf_path = Path(pdf_path)
            base_name = pdf_path.stem
            
            self.logger.info(f"Starting hybrid extraction of {pdf_path}")
            
            hybrid_results = {
                "source_pdf": str(pdf_path),
                "extraction_date": datetime.now().isoformat(),
                "method_used": None,
                "fallback_applied": False,
                "attempts": [],
                "final_result": None,
                "quality_score": 0
            }
            
            if auto_fallback:
                # Step 1: Quick diagnostic to choose initial method
                self.logger.info("Running quick diagnostic...")
                diagnostic_result = self.diagnostic.diagnose_pdf_structure(pdf_path)
                recommended_method = diagnostic_result["recommendations"].get("best_method", "pdfplumber")
                
                hybrid_results["diagnostic_result"] = diagnostic_result
                hybrid_results["recommended_method"] = recommended_method
                
                # Step 2: Try recommended method first
                if recommended_method == "visual_ocr":
                    result = self._try_visual_extraction(pdf_path, hybrid_results)
                elif recommended_method == "pdfplumber":
                    result = self._try_pdfplumber_extraction(pdf_path, hybrid_results)
                else:
                    result = self._try_pdfplumber_extraction(pdf_path, hybrid_results)  # Default to pdfplumber
                
            else:
                # Try pdfplumber first (faster)
                result = self._try_pdfplumber_extraction(pdf_path, hybrid_results)
            
            # Step 3: Fallback if quality is poor
            if auto_fallback and hybrid_results["quality_score"] < 60:
                self.logger.warning(f"Quality score {hybrid_results['quality_score']} is low, applying fallback...")
                hybrid_results["fallback_applied"] = True
                
                if hybrid_results["method_used"] != "visual_ocr":
                    self.logger.info("Falling back to visual OCR extraction...")
                    result = self._try_visual_extraction(pdf_path, hybrid_results)
            
            # Save combined results
            self._save_hybrid_results(hybrid_results, base_name)
            
            self.logger.info(f"Hybrid extraction completed using {hybrid_results['method_used']} (quality: {hybrid_results['quality_score']})")
            return hybrid_results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid extraction: {str(e)}")
            raise
    
    def _try_pdfplumber_extraction(self, pdf_path: Path, hybrid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Try PDFPlumber extraction"""
        try:
            self.logger.info("Attempting PDFPlumber extraction...")
            result = self.alternative_extractor.extract_via_pdfplumber(pdf_path)
            
            quality_score = self._validate_extraction_quality(result["full_text"])
            
            attempt_info = {
                "method": "pdfplumber",
                "success": True,
                "quality_score": quality_score,
                "character_count": len(result["full_text"]),
                "table_count": len(result.get("tables", []))
            }
            
            hybrid_results["attempts"].append(attempt_info)
            hybrid_results["method_used"] = "pdfplumber"
            hybrid_results["final_result"] = result
            hybrid_results["quality_score"] = quality_score
            
            return result
            
        except Exception as e:
            self.logger.error(f"PDFPlumber extraction failed: {str(e)}")
            attempt_info = {
                "method": "pdfplumber",
                "success": False,
                "error": str(e)
            }
            hybrid_results["attempts"].append(attempt_info)
            return {}
    
    def _try_visual_extraction(self, pdf_path: Path, hybrid_results: Dict[str, Any]) -> Dict[str, Any]:
        """Try Visual OCR extraction"""
        try:
            self.logger.info("Attempting Visual OCR extraction...")
            result = self.visual_extractor.extract_via_screenshot(pdf_path)
            
            quality_score = result.get("average_confidence", 0)
            
            attempt_info = {
                "method": "visual_ocr",
                "success": True,
                "quality_score": quality_score,
                "character_count": len(result["full_text"]),
                "average_confidence": quality_score
            }
            
            hybrid_results["attempts"].append(attempt_info)
            hybrid_results["method_used"] = "visual_ocr"
            hybrid_results["final_result"] = result
            hybrid_results["quality_score"] = quality_score
            
            return result
            
        except Exception as e:
            self.logger.error(f"Visual OCR extraction failed: {str(e)}")
            attempt_info = {
                "method": "visual_ocr",
                "success": False,
                "error": str(e)
            }
            hybrid_results["attempts"].append(attempt_info)
            return {}
    
    def _validate_extraction_quality(self, extracted_text: str) -> float:
        """Check if extraction makes sense"""
        if not extracted_text:
            return 0
        
        score = 100
        
        # Check for reasonable character count
        if len(extracted_text) < 100:
            score -= 40
        
        # Check for proper sentence structure
        sentences = len(re.findall(r'[.!?]', extracted_text))
        words = len(extracted_text.split())
        
        if sentences == 0 and words > 10:
            score -= 30
        
        # Check for excessive repeated characters
        repeated_chars = len(re.findall(r'(.)\1{4,}', extracted_text))
        if repeated_chars > 5:
            score -= 20
        
        # Check for common artifacts
        artifacts = ['Col1', 'Col2', 'Col3', '��', 'NULL']
        artifact_count = sum(1 for artifact in artifacts if artifact in extracted_text)
        score -= artifact_count * 10
        
        return max(0, score)
    
    def _save_hybrid_results(self, hybrid_results: Dict[str, Any], base_name: str):
        """Save the hybrid extraction results"""
        # Save best extraction to markdown
        final_result = hybrid_results.get("final_result")
        if final_result:
            best_extraction_path = self.output_dir / f"{base_name}_best_extraction.md"
            with open(best_extraction_path, 'w', encoding='utf-8') as f:
                f.write(f"# Best Extraction Results: {base_name}\n\n")
                f.write(f"**Method Used:** {hybrid_results['method_used']}\n")
                f.write(f"**Quality Score:** {hybrid_results['quality_score']:.1f}\n")
                f.write(f"**Fallback Applied:** {hybrid_results['fallback_applied']}\n\n")
                f.write("---\n\n")
                f.write(final_result.get("full_text", ""))
        
        # Save metadata
        metadata_path = self.output_dir / f"{base_name}_extraction_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(hybrid_results, f, indent=2, ensure_ascii=False)


class ExtractionValidator:
    """Validate extraction quality and completeness"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ExtractionValidator")
    
    def validate_energy_report(self, extracted_text: str) -> Dict[str, Any]:
        """Specific validation for energy reports"""
        required_elements = {
            "account_number": r'\b(?:account|acct)\s*#?\s*\d{6,8}\b',
            "service_address": r'\b\d{3,5}\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Lane|Ln|Drive|Dr|Road|Rd)\b',
            "usage_analysis": r'\b\d+\s*kWh\b',
            "energy_tips": r'\b(?:tip|save|energy|efficient)\b',
            "comparison_data": r'\b(?:more|less|compared?|average|typical)\b'
        }
        
        validation_results = {}
        overall_score = 0
        
        for element, pattern in required_elements.items():
            matches = re.findall(pattern, extracted_text, re.IGNORECASE)
            found = len(matches) > 0
            validation_results[element] = {
                "found": found,
                "matches": matches,
                "count": len(matches)
            }
            if found:
                overall_score += 20
        
        validation_results["overall_score"] = overall_score
        validation_results["completeness"] = overall_score / 100
        
        return validation_results
    
    def calculate_extraction_confidence(self, extraction_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall confidence in extraction"""
        confidence_factors = {}
        
        # OCR confidence (if available)
        if "average_confidence" in extraction_result:
            confidence_factors["ocr_confidence"] = extraction_result["average_confidence"] / 100
        
        # Text structure validation
        text = extraction_result.get("full_text", "")
        if text:
            # Character/word ratios
            words = len(text.split())
            chars = len(text)
            if chars > 0:
                confidence_factors["text_density"] = min(1.0, words * 5 / chars)
            
            # Sentence structure
            sentences = len(re.findall(r'[.!?]', text))
            if words > 0:
                confidence_factors["sentence_structure"] = min(1.0, sentences / (words / 20))
        
        # Overall confidence
        if confidence_factors:
            overall_confidence = np.mean(list(confidence_factors.values()))
        else:
            overall_confidence = 0
        
        confidence_factors["overall_confidence"] = overall_confidence
        
        return confidence_factors


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup comprehensive logging"""
    # Create logs directory
    log_dir = Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging configuration
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file or log_dir / "visual_extraction.log", encoding='utf-8')
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('fitz').setLevel(logging.WARNING)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Visual PDF Extraction System - Extract text using screenshot + OCR and intelligent fallbacks"
    )
    
    parser.add_argument(
        "pdf_path",
        help="Path to PDF file or directory"
    )
    
    parser.add_argument(
        "--method",
        choices=["screenshot", "pdfplumber", "hybrid", "diagnose"],
        default="hybrid",
        help="Extraction method to use (default: hybrid)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for screenshot conversion (default: 300)"
    )
    
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Apply image preprocessing for better OCR"
    )
    
    parser.add_argument(
        "--auto-fallback",
        action="store_true",
        default=True,
        help="Automatically fallback to other methods if quality is poor"
    )
    
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all extraction methods (diagnostic mode)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process multiple PDFs in parallel"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process PDFs in subdirectories recursively"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize extractors
        if args.method == "screenshot":
            extractor = VisualPDFExtractor(dpi=args.dpi, output_dir=f"{args.output_dir}/visual_extraction")
        elif args.method == "pdfplumber":
            extractor = AlternativePDFExtractor(output_dir=f"{args.output_dir}/visual_extraction")
        elif args.method == "hybrid":
            extractor = HybridPDFExtractor(output_dir=f"{args.output_dir}/combined")
        elif args.method == "diagnose":
            extractor = PDFExtractionDiagnostic(output_dir=f"{args.output_dir}/diagnostic")
        
        pdf_path = Path(args.pdf_path)
        
        if pdf_path.is_file() and pdf_path.suffix.lower() == '.pdf':
            # Single PDF processing
            logger.info(f"Processing single PDF: {pdf_path}")
            
            if args.method == "screenshot":
                result = extractor.extract_via_screenshot(str(pdf_path), preprocess=args.preprocess)
            elif args.method == "pdfplumber":
                result = extractor.extract_via_pdfplumber(str(pdf_path))
            elif args.method == "hybrid":
                result = extractor.extract_with_fallback(str(pdf_path), auto_fallback=args.auto_fallback)
            elif args.method == "diagnose":
                result = extractor.diagnose_pdf_structure(str(pdf_path))
            
            logger.info("Extraction completed successfully!")
            
            # Print summary
            if args.method == "screenshot":
                print(f"\n✅ Visual OCR extraction completed!")
                print(f"   Average confidence: {result.get('average_confidence', 0):.1f}%")
                print(f"   Total pages: {result.get('total_pages', 0)}")
                print(f"   Total characters: {len(result.get('full_text', ''))}")
            elif args.method == "hybrid":
                print(f"\n✅ Hybrid extraction completed!")
                print(f"   Method used: {result.get('method_used', 'unknown')}")
                print(f"   Quality score: {result.get('quality_score', 0):.1f}")
                print(f"   Fallback applied: {result.get('fallback_applied', False)}")
            elif args.method == "diagnose":
                print(f"\n🔍 Diagnostic analysis completed!")
                print(f"   Recommended method: {result.get('recommendations', {}).get('best_method', 'unknown')}")
                print(f"   Best score: {result.get('recommendations', {}).get('best_score', 0):.1f}")
            
        elif pdf_path.is_dir():
            # Directory processing
            logger.info(f"Processing directory: {pdf_path}")
            
            # Find PDF files
            if args.recursive:
                pdf_files = list(pdf_path.rglob("*.pdf"))
            else:
                pdf_files = list(pdf_path.glob("*.pdf"))
            
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            if args.parallel and len(pdf_files) > 1:
                # Parallel processing
                logger.info("Processing PDFs in parallel...")
                # Implementation would go here
                logger.warning("Parallel processing not implemented yet - processing sequentially")
            
            # Sequential processing
            for pdf_file in pdf_files:
                logger.info(f"Processing: {pdf_file}")
                try:
                    if args.method == "screenshot":
                        result = extractor.extract_via_screenshot(str(pdf_file), preprocess=args.preprocess)
                    elif args.method == "pdfplumber":
                        result = extractor.extract_via_pdfplumber(str(pdf_file))
                    elif args.method == "hybrid":
                        result = extractor.extract_with_fallback(str(pdf_file), auto_fallback=args.auto_fallback)
                    elif args.method == "diagnose":
                        result = extractor.diagnose_pdf_structure(str(pdf_file))
                    
                    logger.info(f"✅ Completed: {pdf_file}")
                except Exception as e:
                    logger.error(f"❌ Failed: {pdf_file} - {str(e)}")
        
        else:
            logger.error(f"Invalid path: {pdf_path}")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Extraction cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()