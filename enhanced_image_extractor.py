#!/usr/bin/env python3
"""
Enhanced PDF Image Extractor with Advanced Analysis

This script provides comprehensive PDF image extraction with:
- Advanced image visibility detection and classification
- Text-image correlation and context analysis  
- Embedding-optimized image processing
- Comprehensive metadata generation with text context
- Integration with enhanced text extraction

Features:
- Detect visible vs embedded vs background images
- Correlate images with surrounding text content
- Enhanced classification using text context
- Optimize images for vision-language models
- Generate detailed correlation analysis
- Create multimodal embedding pairs

Author: Assistant
Date: 2025-09-02
"""

import os
import sys
import json
import pathlib
import argparse
import logging
import hashlib
import io
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import math
import re

try:
    import pymupdf as fitz
except ImportError:
    print("Error: PyMuPDF is not installed.")
    print("Please install it using: pip install PyMuPDF")
    sys.exit(1)

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageStat
except ImportError:
    print("Error: Pillow is not installed.")
    print("Please install it using: pip install Pillow")
    sys.exit(1)

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. Text detection features will be disabled.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Advanced image analysis features will be disabled.")


class ImageVisibilityAnalyzer:
    """Analyzes image visibility and context within PDF pages."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_image_visibility(self, image_data: Dict[str, Any], page_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine if image is visible, embedded, or background.
        
        Args:
            image_data (Dict): Image metadata including coordinates and dimensions
            page_content (Dict): Page text content and structure information
        
        Returns:
            Dict: Visibility analysis results
        """
        coordinates = image_data.get("coordinates", {})
        dimensions = image_data.get("dimensions", {})
        page_width = page_content.get("page_width", 612)  # Default US Letter width
        page_height = page_content.get("page_height", 792)  # Default US Letter height
        
        # Calculate position ratios
        x_ratio = coordinates.get("x", 0) / page_width if page_width > 0 else 0
        y_ratio = coordinates.get("y", 0) / page_height if page_height > 0 else 0
        width_ratio = dimensions.get("width", 0) / page_width if page_width > 0 else 0
        height_ratio = dimensions.get("height", 0) / page_height if page_height > 0 else 0
        
        # Analysis factors
        factors = {
            "size_factor": self._analyze_size_factor(width_ratio, height_ratio),
            "position_factor": self._analyze_position_factor(x_ratio, y_ratio),
            "overlap_factor": self._analyze_text_overlap(coordinates, dimensions, page_content),
            "aspect_factor": self._analyze_aspect_ratio(dimensions),
            "content_factor": self._analyze_image_content(image_data)
        }
        
        # Determine visibility
        visibility_score = self._calculate_visibility_score(factors)
        visibility_type = self._classify_visibility(visibility_score, factors)
        
        return {
            "visibility": visibility_type,
            "confidence": round(visibility_score, 2),
            "detection_method": "coordinate_analysis",
            "reasoning": self._generate_visibility_reasoning(factors, visibility_type),
            "analysis_factors": factors,
            "position_ratios": {
                "x_ratio": round(x_ratio, 3),
                "y_ratio": round(y_ratio, 3), 
                "width_ratio": round(width_ratio, 3),
                "height_ratio": round(height_ratio, 3)
            }
        }
    
    def _analyze_size_factor(self, width_ratio: float, height_ratio: float) -> Dict[str, float]:
        """Analyze image size relative to page."""
        area_ratio = width_ratio * height_ratio
        
        return {
            "area_ratio": area_ratio,
            "is_large": 1.0 if area_ratio > 0.3 else area_ratio / 0.3,
            "is_small": 1.0 if area_ratio < 0.05 else (0.05 - area_ratio) / 0.05,
            "is_page_sized": 1.0 if area_ratio > 0.8 else 0.0
        }
    
    def _analyze_position_factor(self, x_ratio: float, y_ratio: float) -> Dict[str, float]:
        """Analyze image position on page."""
        return {
            "is_header": 1.0 if y_ratio < 0.15 else 0.0,
            "is_footer": 1.0 if y_ratio > 0.85 else 0.0,
            "is_margin": 1.0 if (x_ratio < 0.1 or x_ratio > 0.9) else 0.0,
            "is_centered": 1.0 if (0.4 < x_ratio < 0.6 and 0.3 < y_ratio < 0.7) else 0.0
        }
    
    def _analyze_text_overlap(self, coordinates: Dict[str, int], dimensions: Dict[str, int], 
                            page_content: Dict[str, Any]) -> Dict[str, float]:
        """Analyze overlap with text content."""
        # Simplified analysis - in production would use actual text block coordinates
        text_density = len(page_content.get("text", "").split()) / max(page_content.get("char_count", 1), 1)
        
        return {
            "text_density": text_density,
            "likely_overlapping": 1.0 if text_density > 0.1 else text_density * 10,
            "isolated": 1.0 if text_density < 0.05 else 0.0
        }
    
    def _analyze_aspect_ratio(self, dimensions: Dict[str, int]) -> Dict[str, float]:
        """Analyze image aspect ratio."""
        width = dimensions.get("width", 1)
        height = dimensions.get("height", 1)
        aspect_ratio = width / height
        
        return {
            "aspect_ratio": aspect_ratio,
            "is_square": 1.0 if 0.8 <= aspect_ratio <= 1.2 else 0.0,
            "is_wide": 1.0 if aspect_ratio > 2.0 else 0.0,
            "is_tall": 1.0 if aspect_ratio < 0.5 else 0.0
        }
    
    def _analyze_image_content(self, image_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze image content characteristics."""
        color_palette = image_data.get("color_palette", 100)
        text_detected = image_data.get("text_detected", False)
        
        return {
            "has_text": 1.0 if text_detected else 0.0,
            "color_richness": min(color_palette / 1000, 1.0),
            "likely_graphic": 1.0 if color_palette < 50 else 0.0,
            "likely_photo": 1.0 if color_palette > 500 and not text_detected else 0.0
        }
    
    def _calculate_visibility_score(self, factors: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall visibility score."""
        score = 0.5  # Base score
        
        size_f = factors["size_factor"]
        pos_f = factors["position_factor"]
        overlap_f = factors["overlap_factor"]
        aspect_f = factors["aspect_factor"]
        content_f = factors["content_factor"]
        
        # Visible indicators
        if size_f["is_large"] > 0.5:
            score += 0.2
        if pos_f["is_centered"] > 0.5:
            score += 0.15
        if content_f["likely_photo"] > 0.5:
            score += 0.1
        
        # Background indicators
        if size_f["is_page_sized"] > 0.5:
            score -= 0.3
        if overlap_f["text_density"] > 0.5:
            score -= 0.2
        
        # Embedded indicators
        if size_f["is_small"] > 0.7:
            score += 0.1  # Small can be visible (logos) or embedded
        if pos_f["is_header"] > 0.5 or pos_f["is_footer"] > 0.5:
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _classify_visibility(self, score: float, factors: Dict[str, Dict[str, float]]) -> str:
        """Classify visibility type based on score and factors."""
        size_f = factors["size_factor"]
        
        # Background detection
        if size_f["is_page_sized"] > 0.5:
            return "background"
        
        # Visibility classification
        if score > 0.7:
            return "visible"
        elif score > 0.4:
            return "embedded"
        else:
            return "background"
    
    def _generate_visibility_reasoning(self, factors: Dict[str, Dict[str, float]], 
                                     visibility_type: str) -> str:
        """Generate human-readable reasoning for visibility classification."""
        reasons = []
        
        size_f = factors["size_factor"]
        pos_f = factors["position_factor"]
        content_f = factors["content_factor"]
        
        if size_f["is_large"] > 0.5:
            reasons.append("large size")
        elif size_f["is_small"] > 0.5:
            reasons.append("small size")
        
        if pos_f["is_centered"] > 0.5:
            reasons.append("centered position")
        if pos_f["is_header"] > 0.5:
            reasons.append("header position")
        if pos_f["is_footer"] > 0.5:
            reasons.append("footer position")
        
        if content_f["likely_photo"] > 0.5:
            reasons.append("photographic content")
        if content_f["has_text"] > 0.5:
            reasons.append("contains text")
        
        if size_f["is_page_sized"] > 0.5:
            reasons.append("page-sized dimensions")
        
        return ", ".join(reasons) if reasons else "general characteristics"


class TextImageCorrelator:
    """Correlates images with surrounding text content."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Keywords for different image types and their contexts
        self.context_keywords = {
            "energy_tips": ["tip", "advice", "save", "energy", "reduce", "efficient", "laundry", "loads"],
            "usage_data": ["usage", "kwh", "typical", "average", "month", "bill", "consumption"],
            "contact_info": ["contact", "call", "visit", "website", "scan", "qr", "code"],
            "account_info": ["account", "number", "service", "address", "customer"],
            "savings": ["save", "savings", "money", "cost", "reduce", "lower"],
            "comparison": ["above", "below", "typical", "average", "compare", "neighbors"]
        }
    
    def correlate_with_text(self, images: List[Dict[str, Any]], 
                          text_content: str, 
                          page_text_blocks: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """
        Create detailed image-text correlations.
        
        Args:
            images (List[Dict]): Image metadata
            text_content (str): Full text content
            page_text_blocks (List[Dict]): Detailed text block information
        
        Returns:
            List[Dict]: Enhanced images with text correlations
        """
        enhanced_images = []
        
        for img in images:
            correlation_data = self._analyze_image_text_correlation(img, text_content, page_text_blocks)
            
            # Add correlation data to image
            enhanced_img = img.copy()
            enhanced_img["text_correlation"] = correlation_data
            enhanced_img["contextual_description"] = self._generate_contextual_description(img, correlation_data)
            
            enhanced_images.append(enhanced_img)
        
        return enhanced_images
    
    def _analyze_image_text_correlation(self, image: Dict[str, Any], 
                                      text_content: str, 
                                      page_text_blocks: Optional[List[Dict]]) -> Dict[str, Any]:
        """Analyze correlation between specific image and text."""
        img_type = image.get("classification", {}).get("type", "unknown")
        page_num = image.get("page_number", 0)
        coordinates = image.get("coordinates", {})
        
        # Extract relevant text sections
        nearby_text = self._extract_nearby_text(text_content, page_num, coordinates)
        
        # Calculate correlation scores for different contexts
        context_scores = {}
        for context, keywords in self.context_keywords.items():
            score = self._calculate_context_score(nearby_text, keywords)
            if score > 0.1:  # Only include relevant contexts
                context_scores[context] = score
        
        # Find best matching context
        best_context = max(context_scores.items(), key=lambda x: x[1]) if context_scores else ("general", 0.1)
        
        # Analyze specific correlations based on image type
        type_specific = self._analyze_type_specific_correlation(img_type, nearby_text, text_content)
        
        return {
            "nearby_text": nearby_text,
            "context_scores": context_scores,
            "primary_context": best_context[0],
            "context_confidence": round(best_context[1], 2),
            "type_specific_analysis": type_specific,
            "correlation_strength": self._calculate_overall_correlation(context_scores, type_specific),
            "supporting_relationships": self._identify_supporting_relationships(image, nearby_text)
        }
    
    def _extract_nearby_text(self, text_content: str, page_num: int, 
                           coordinates: Dict[str, int]) -> str:
        """Extract text content likely related to the image."""
        # Split text by pages (simplified - in production would use actual page boundaries)
        pages = text_content.split('\n\n')
        
        if page_num < len(pages):
            page_text = pages[page_num]
        else:
            page_text = text_content
        
        # Extract paragraphs around image location
        paragraphs = page_text.split('\n\n')
        
        # For energy reports, look for specific patterns
        relevant_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Include paragraphs with energy-related content
            if any(keyword in para.lower() for keyword in 
                   ["energy", "tip", "save", "usage", "kwh", "laundry", "load"]):
                relevant_paragraphs.append(para)
        
        # If no specific matches, return surrounding text
        if not relevant_paragraphs:
            mid_point = len(paragraphs) // 2
            start_idx = max(0, mid_point - 1)
            end_idx = min(len(paragraphs), mid_point + 2)
            relevant_paragraphs = paragraphs[start_idx:end_idx]
        
        return " ".join(relevant_paragraphs[:3])  # Limit to 3 paragraphs
    
    def _calculate_context_score(self, text: str, keywords: List[str]) -> float:
        """Calculate how well text matches a specific context."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        word_count = len(text_lower.split())
        
        if word_count == 0:
            return 0.0
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Calculate score based on keyword density
        keyword_density = matches / len(keywords)
        text_relevance = min(matches / max(word_count * 0.1, 1), 1.0)
        
        return (keyword_density + text_relevance) / 2
    
    def _analyze_type_specific_correlation(self, img_type: str, nearby_text: str, 
                                         full_text: str) -> Dict[str, Any]:
        """Analyze correlation specific to image type."""
        analysis = {"type": img_type, "specific_indicators": []}
        
        nearby_lower = nearby_text.lower()
        full_lower = full_text.lower()
        
        if img_type == "logo":
            # Look for company/brand references
            if any(brand in full_lower for brand in ["xcel", "energy", "franklin"]):
                analysis["specific_indicators"].append("company_branding")
            if "franklinenergy.com" in nearby_lower or "website" in nearby_lower:
                analysis["specific_indicators"].append("website_reference")
        
        elif img_type == "photo":
            # Look for activity references
            if "laundry" in nearby_lower and any(word in nearby_lower for word in ["load", "full", "tip"]):
                analysis["specific_indicators"].append("laundry_activity_reference")
            if "save" in nearby_lower and "energy" in nearby_lower:
                analysis["specific_indicators"].append("energy_saving_context")
        
        elif img_type == "chart":
            # Look for data references
            if any(word in nearby_lower for word in ["usage", "kwh", "typical", "above", "below"]):
                analysis["specific_indicators"].append("usage_data_reference")
            if "%" in nearby_text or "percent" in nearby_lower:
                analysis["specific_indicators"].append("percentage_data")
        
        elif img_type == "unknown":
            # Try to identify from context
            if "qr" in nearby_lower or "scan" in nearby_lower or "code" in nearby_lower:
                analysis["likely_type"] = "qr_code"
                analysis["specific_indicators"].append("qr_code_reference")
        
        analysis["confidence"] = len(analysis["specific_indicators"]) * 0.2
        return analysis
    
    def _calculate_overall_correlation(self, context_scores: Dict[str, float], 
                                     type_specific: Dict[str, Any]) -> float:
        """Calculate overall correlation strength."""
        # Base score from context
        context_score = max(context_scores.values()) if context_scores else 0.1
        
        # Bonus from type-specific analysis
        type_bonus = type_specific.get("confidence", 0.0) * 0.3
        
        # Final correlation strength
        return min(context_score + type_bonus, 1.0)
    
    def _identify_supporting_relationships(self, image: Dict[str, Any], 
                                         nearby_text: str) -> List[str]:
        """Identify specific supporting relationships between image and text."""
        relationships = []
        
        img_type = image.get("classification", {}).get("type", "unknown")
        text_lower = nearby_text.lower()
        
        # Direct reference relationships
        if "image" in text_lower or "picture" in text_lower or "photo" in text_lower:
            relationships.append("direct_reference")
        
        # Instructional relationships
        if img_type in ["photo", "illustration"] and any(word in text_lower for word in ["tip", "advice", "how to"]):
            relationships.append("instructional_support")
        
        # Data visualization relationships
        if img_type == "chart" and any(word in text_lower for word in ["usage", "data", "above", "typical"]):
            relationships.append("data_visualization")
        
        # Branding relationships
        if img_type == "logo" and any(word in text_lower for word in ["company", "energy", "contact"]):
            relationships.append("brand_identification")
        
        # Action relationships
        if "scan" in text_lower or "visit" in text_lower:
            relationships.append("call_to_action")
        
        return relationships
    
    def _generate_contextual_description(self, image: Dict[str, Any], 
                                       correlation_data: Dict[str, Any]) -> str:
        """Generate human-readable contextual description."""
        img_type = image.get("classification", {}).get("type", "unknown")
        primary_context = correlation_data.get("primary_context", "general")
        nearby_text = correlation_data.get("nearby_text", "")
        
        # Base description
        descriptions = {
            "logo": "Company logo or branding element",
            "photo": "Photograph or illustration",
            "chart": "Chart or data visualization",
            "diagram": "Diagram or schematic",
            "unknown": "Unidentified image element"
        }
        
        base_desc = descriptions.get(img_type, "Image element")
        
        # Context-specific enhancements
        if primary_context == "energy_tips" and img_type == "photo":
            return "Illustration supporting energy-saving tip about laundry practices"
        elif primary_context == "contact_info" and img_type in ["unknown", "diagram"]:
            return "QR code for accessing online energy account or website"
        elif primary_context == "usage_data" and img_type == "chart":
            return "Energy usage comparison chart showing consumption relative to typical usage"
        elif primary_context == "account_info" and img_type == "logo":
            return "Company logo associated with energy service provider branding"
        
        # Fallback with context
        context_text = primary_context.replace("_", " ").title()
        return f"{base_desc} related to {context_text.lower()}"


class EnhancedImageProcessor:
    """Enhanced image processing for embedding optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def enhance_for_embeddings(self, image: Image.Image, 
                             enhancement_profile: str = "embedding_optimized") -> Image.Image:
        """
        Optimize images specifically for vision-language models.
        
        Args:
            image (Image.Image): Input image
            enhancement_profile (str): Enhancement profile
        
        Returns:
            Image.Image: Optimized image
        """
        try:
            if enhancement_profile == "embedding_optimized":
                return self._embedding_optimization(image)
            elif enhancement_profile == "ocr_ready":
                return self._ocr_optimization(image)
            elif enhancement_profile == "analysis_ready":
                return self._analysis_optimization(image)
            else:
                return image
        
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {str(e)}")
            return image
    
    def _embedding_optimization(self, image: Image.Image) -> Image.Image:
        """Optimize for vision-language model embeddings."""
        # Standardize size for consistent processing
        target_size = (224, 224)  # Common vision model input size
        
        # Maintain aspect ratio with padding if needed
        img_ratio = image.width / image.height
        target_ratio = target_size[0] / target_size[1]
        
        if img_ratio > target_ratio:
            # Image is wider
            new_width = target_size[0]
            new_height = int(target_size[0] / img_ratio)
        else:
            # Image is taller
            new_height = target_size[1]
            new_width = int(target_size[1] * img_ratio)
        
        # Resize with high-quality resampling
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create padded image if needed
        if (new_width, new_height) != target_size:
            padded = Image.new('RGB', target_size, (255, 255, 255))
            paste_x = (target_size[0] - new_width) // 2
            paste_y = (target_size[1] - new_height) // 2
            padded.paste(resized, (paste_x, paste_y))
            enhanced = padded
        else:
            enhanced = resized
        
        # Normalize colors
        enhancer = ImageEnhance.Color(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        # Slight sharpening
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        return enhanced
    
    def _ocr_optimization(self, image: Image.Image) -> Image.Image:
        """Optimize for OCR text recognition."""
        # Convert to grayscale for better OCR
        if image.mode != 'L':
            enhanced = image.convert('L')
        else:
            enhanced = image.copy()
        
        # Increase contrast
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(2.0)
        
        # Sharpen for better text clarity
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.5)
        
        # Apply noise reduction
        enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
        
        return enhanced
    
    def _analysis_optimization(self, image: Image.Image) -> Image.Image:
        """Optimize for general image analysis."""
        enhanced = image.copy()
        
        # Improve overall clarity
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        # Color normalization
        if enhanced.mode in ('RGB', 'RGBA'):
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.05)
        
        return enhanced
    
    def analyze_image_characteristics(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze advanced image characteristics."""
        characteristics = {}
        
        try:
            # Basic statistics
            if NUMPY_AVAILABLE:
                img_array = np.array(image)
                characteristics["mean_brightness"] = float(np.mean(img_array))
                characteristics["std_brightness"] = float(np.std(img_array))
            
            # Color analysis using PIL
            if image.mode in ('RGB', 'RGBA'):
                stat = ImageStat.Stat(image)
                characteristics["color_means"] = stat.mean
                characteristics["color_stddevs"] = stat.stddev
                
                # Dominant color detection (simplified)
                colors = image.getcolors(maxcolors=256*256*256)
                if colors:
                    dominant_color = max(colors, key=lambda x: x[0])
                    characteristics["dominant_color"] = {
                        "count": dominant_color[0],
                        "rgb": dominant_color[1] if isinstance(dominant_color[1], tuple) else [dominant_color[1]]
                    }
            
            # Edge detection indicators
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image
            
            # Simple edge detection using filter
            edges = gray_image.filter(ImageFilter.FIND_EDGES)
            edge_pixels = sum(1 for pixel in edges.getdata() if pixel > 30)
            total_pixels = edges.width * edges.height
            characteristics["edge_density"] = edge_pixels / total_pixels if total_pixels > 0 else 0
            
        except Exception as e:
            self.logger.debug(f"Error analyzing image characteristics: {str(e)}")
        
        return characteristics


class EnhancedPDFImageExtractor:
    """
    Enhanced PDF image extractor with advanced visibility detection and correlation.
    """
    
    def __init__(self, 
                 output_dir: str = "output", 
                 enable_enhancement: bool = True,
                 log_level: int = logging.INFO):
        """Initialize the Enhanced PDF Image Extractor."""
        # Initialize base directories
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create enhanced directory structure
        self.images_dir = self.output_dir / "images"
        self.enhanced_dir = self.output_dir / "enhanced"
        self.metadata_dir = self.output_dir / "metadata"
        self.logs_dir = self.output_dir / "logs"
        
        # New directories for enhanced features
        self.visible_dir = self.images_dir / "visible"
        self.embedded_dir = self.images_dir / "embedded"
        self.by_type_dir = self.images_dir / "by_type"
        
        for dir_path in [self.images_dir, self.enhanced_dir, self.metadata_dir, self.logs_dir,
                        self.visible_dir, self.embedded_dir, self.by_type_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.logs_dir / "enhanced_extraction.log"
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.visibility_analyzer = ImageVisibilityAnalyzer()
        self.text_correlator = TextImageCorrelator()
        self.image_processor = EnhancedImageProcessor()
        self.enable_enhancement = enable_enhancement
        
        # Image hash storage for duplicate detection
        self.image_hashes = {}
        
        # Enhanced classification confidence thresholds
        self.classification_thresholds = {
            'logo': 0.6,
            'chart': 0.7,
            'diagram': 0.6,
            'photo': 0.5,
            'qr_code': 0.8
        }
    
    def extract_images_enhanced(self, 
                              pdf_path: str, 
                              text_content: Optional[str] = None,
                              filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract images with enhanced analysis and text correlation.
        
        Args:
            pdf_path (str): Path to PDF file
            text_content (str): Associated text content for correlation
            filters (Dict): Filter configuration
        
        Returns:
            Dict: Enhanced extraction results
        """
        pdf_path = pathlib.Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Starting enhanced image extraction from: {pdf_path}")
        
        try:
            # Open PDF
            pdf_doc = fitz.open(str(pdf_path))
            
            if pdf_doc.is_encrypted:
                self.logger.warning("PDF is encrypted. Attempting to open...")
                if not pdf_doc.authenticate(""):
                    raise ValueError("PDF requires password authentication")
            
            extraction_results = {
                "source_pdf": pdf_path.name,
                "extraction_date": datetime.now().isoformat(),
                "total_pages": len(pdf_doc),
                "extracted_images": [],
                "visibility_analysis": {},
                "text_correlation": {},
                "processing_summary": {}
            }
            
            total_images = 0
            page_contents = {}
            
            # Process each page
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                page_rect = page.rect
                
                # Get page text for correlation
                page_text = page.get_text()
                page_contents[page_num] = {
                    "text": page_text,
                    "page_width": page_rect.width,
                    "page_height": page_rect.height,
                    "char_count": len(page_text)
                }
                
                # Extract images from page
                page_images = self._extract_images_from_page_enhanced(
                    page, page_num, pdf_path.stem, page_contents[page_num]
                )
                
                if page_images:
                    extraction_results["extracted_images"].extend(page_images)
                    total_images += len(page_images)
                    self.logger.debug(f"Extracted {len(page_images)} images from page {page_num}")
            
            pdf_doc.close()
            
            # Apply filters if specified
            if filters:
                extraction_results["extracted_images"] = self.apply_filters_enhanced(
                    extraction_results["extracted_images"], filters
                )
            
            # Correlate with text content if provided
            if text_content:
                extraction_results["extracted_images"] = self.text_correlator.correlate_with_text(
                    extraction_results["extracted_images"], text_content
                )
            
            # Generate processing summary
            extraction_results["processing_summary"] = self._generate_processing_summary(
                extraction_results["extracted_images"]
            )
            
            self.logger.info(f"Successfully extracted {total_images} images with enhanced analysis")
            return extraction_results
            
        except Exception as e:
            self.logger.error(f"Error during enhanced extraction: {str(e)}")
            raise
    
    def _extract_images_from_page_enhanced(self, 
                                         page: fitz.Page, 
                                         page_num: int, 
                                         pdf_name: str,
                                         page_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced image extraction from a single page."""
        page_images = []
        image_list = page.get_images()
        
        if not image_list:
            return page_images
        
        for img_index, img_ref in enumerate(image_list):
            try:
                # Get image data
                xref = img_ref[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                # Skip if image is too small or invalid
                if pix.width < 10 or pix.height < 10:
                    pix = None
                    continue
                
                # Convert to RGB if necessary
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                else:  # CMYK: convert to RGB first
                    pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                    img_data = pix_rgb.tobytes("png")
                    pix_rgb = None
                
                # Create PIL Image for processing
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Get image location on page
                image_rects = page.get_image_rects(img_ref)
                if image_rects:
                    rect = image_rects[0]
                    coordinates = {
                        "x": int(rect.x0),
                        "y": int(rect.y0),
                        "width": int(rect.width),
                        "height": int(rect.height)
                    }
                else:
                    coordinates = {
                        "x": 0,
                        "y": 0,
                        "width": pix.width,
                        "height": pix.height
                    }
                
                # Basic image metadata
                basic_metadata = {
                    "page_number": page_num,
                    "image_index": img_index,
                    "coordinates": coordinates,
                    "dimensions": {
                        "width": pil_image.width,
                        "height": pil_image.height
                    },
                    "original_format": "PNG",
                    "size_bytes": len(img_data),
                    "aspect_ratio": round(pil_image.width / pil_image.height, 2)
                }
                
                # Enhanced analysis
                
                # 1. Visibility analysis
                visibility_result = self.visibility_analyzer.analyze_image_visibility(
                    basic_metadata, page_content
                )
                
                # 2. Enhanced classification (using original classify_image method plus enhancements)
                classification = self.classify_image_enhanced(pil_image, page_content, coordinates, page_content.get("text", ""))
                
                # 3. Advanced image characteristics
                characteristics = self.image_processor.analyze_image_characteristics(pil_image)
                
                # 4. Text detection
                text_detected = self._detect_text_in_image(pil_image) if TESSERACT_AVAILABLE else False
                
                # 5. Calculate image hash
                img_hash = self._calculate_image_hash(pil_image)
                is_duplicate = img_hash in self.image_hashes
                if not is_duplicate:
                    self.image_hashes[img_hash] = f"{pdf_name}_page{page_num}_img{img_index}"
                
                # Generate filename based on enhanced classification
                filename = f"{pdf_name}_page{page_num}_img{img_index}_{classification['type']}.jpg"
                
                # Create comprehensive metadata
                metadata = {
                    "filename": filename,
                    **basic_metadata,
                    "classification": classification,
                    "visibility_analysis": visibility_result,
                    "characteristics": characteristics,
                    "text_detected": text_detected,
                    "hash": img_hash,
                    "is_duplicate": is_duplicate,
                    "enhancement_applied": False
                }
                
                # Save original image to appropriate directory
                if not is_duplicate:
                    self._save_image_organized(pil_image, metadata, visibility_result["visibility"], classification["type"])
                    
                    # Apply enhancement if enabled
                    if self.enable_enhancement:
                        enhanced_image = self.image_processor.enhance_for_embeddings(pil_image, "embedding_optimized")
                        enhanced_filename = filename.replace(".jpg", "_enhanced.jpg")
                        enhanced_path = self.enhanced_dir / enhanced_filename
                        enhanced_image.save(enhanced_path, "JPEG", quality=95)
                        metadata["enhancement_applied"] = True
                        metadata["enhanced_filename"] = enhanced_filename
                
                page_images.append(metadata)
                
                # Cleanup
                pix = None
                
            except Exception as e:
                self.logger.warning(f"Failed to process image {img_index} on page {page_num}: {str(e)}")
                continue
        
        return page_images
    
    def classify_image_enhanced(self, 
                              image: Image.Image, 
                              page_info: Dict[str, Any], 
                              coordinates: Dict[str, int],
                              page_text: str) -> Dict[str, Any]:
        """Enhanced image classification using text context."""
        # Start with basic classification (similar to original but enhanced)
        width, height = image.size
        aspect_ratio = width / height
        page_width = page_info.get("page_width", 612)
        page_height = page_info.get("page_height", 792)
        
        # Position analysis
        x_pos = coordinates["x"] / page_width if page_width > 0 else 0
        y_pos = coordinates["y"] / page_height if page_height > 0 else 0
        width_ratio = coordinates["width"] / page_width if page_width > 0 else 0
        height_ratio = coordinates["height"] / page_height if page_height > 0 else 0
        
        # Size analysis
        area = width * height
        is_small = area < 10000
        is_large = width_ratio > 0.6 or height_ratio > 0.6
        
        # Color analysis
        colors = image.getcolors(maxcolors=256*256*256)
        unique_colors = len(colors) if colors else 1000
        is_limited_palette = unique_colors < 50
        
        # Text detection
        has_text = self._detect_text_in_image(image) if TESSERACT_AVAILABLE else False
        
        # Enhanced text context analysis
        text_context = self._analyze_text_context(page_text, coordinates)
        
        # Classification scores with text context enhancement
        scores = {
            'logo': 0.0,
            'chart': 0.0,
            'diagram': 0.0,
            'photo': 0.0,
            'qr_code': 0.0
        }
        
        # Logo indicators
        if is_small:
            scores['logo'] += 0.4
        if y_pos < 0.2 or y_pos > 0.8:  # Header or footer
            scores['logo'] += 0.3
        if x_pos < 0.2 or x_pos > 0.8:  # Left or right margin
            scores['logo'] += 0.2
        if is_limited_palette:
            scores['logo'] += 0.3
        if text_context.get("has_company_references", False):
            scores['logo'] += 0.2
        
        # Chart indicators
        if is_large:
            scores['chart'] += 0.3
        if has_text:
            scores['chart'] += 0.4
        if is_limited_palette:
            scores['chart'] += 0.3
        if text_context.get("has_data_references", False):
            scores['chart'] += 0.3
        
        # QR Code indicators (enhanced detection)
        if 0.8 <= aspect_ratio <= 1.2:  # Square
            scores['qr_code'] += 0.3
        if is_small and is_limited_palette:
            scores['qr_code'] += 0.2
        if text_context.get("has_qr_references", False):
            scores['qr_code'] += 0.4
        if has_text and "franklinenergy" in page_text.lower():
            scores['qr_code'] += 0.2
        
        # Photo indicators
        if unique_colors > 1000:
            scores['photo'] += 0.4
        if not has_text:
            scores['photo'] += 0.3
        if area > 100000:
            scores['photo'] += 0.3
        if text_context.get("has_activity_references", False):
            scores['photo'] += 0.3
        
        # Diagram indicators
        if is_limited_palette:
            scores['diagram'] += 0.4
        if has_text:
            scores['diagram'] += 0.3
        if 0.5 <= aspect_ratio <= 2.0:
            scores['diagram'] += 0.2
        
        # Determine final classification
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type], 1.0)
        
        # Apply minimum confidence thresholds
        min_confidence = self.classification_thresholds.get(best_type, 0.5)
        if confidence < min_confidence:
            best_type = "unknown"
            confidence = 0.3
        
        # Generate enhanced reasoning
        reasoning_parts = []
        if is_small:
            reasoning_parts.append("small size")
        if is_large:
            reasoning_parts.append("large size")
        if y_pos < 0.2:
            reasoning_parts.append("header position")
        elif y_pos > 0.8:
            reasoning_parts.append("footer position")
        if has_text:
            reasoning_parts.append("contains text")
        if text_context.get("context_type"):
            reasoning_parts.append(f"{text_context['context_type']} context")
        
        reasoning = ", ".join(reasoning_parts) if reasoning_parts else "general characteristics"
        
        return {
            "type": best_type,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
            "scores": {k: round(v, 2) for k, v in scores.items()},
            "text_context_analysis": text_context
        }
    
    def _analyze_text_context(self, page_text: str, coordinates: Dict[str, int]) -> Dict[str, Any]:
        """Analyze surrounding text context for better classification."""
        text_lower = page_text.lower()
        
        context_analysis = {
            "has_company_references": False,
            "has_data_references": False,
            "has_qr_references": False,
            "has_activity_references": False,
            "context_type": None
        }
        
        # Company references
        if any(word in text_lower for word in ["xcel", "franklin", "energy", "company"]):
            context_analysis["has_company_references"] = True
        
        # Data references
        if any(word in text_lower for word in ["usage", "kwh", "typical", "above", "below", "%"]):
            context_analysis["has_data_references"] = True
            context_analysis["context_type"] = "usage_data"
        
        # QR code references
        if any(word in text_lower for word in ["scan", "qr", "code", "franklinenergy.com", "visit"]):
            context_analysis["has_qr_references"] = True
            context_analysis["context_type"] = "qr_code"
        
        # Activity references (for photos)
        if any(word in text_lower for word in ["laundry", "tip", "save", "loads", "full"]):
            context_analysis["has_activity_references"] = True
            context_analysis["context_type"] = "energy_tip"
        
        return context_analysis
    
    def _save_image_organized(self, image: Image.Image, metadata: Dict[str, Any], 
                            visibility: str, img_type: str):
        """Save image in organized directory structure."""
        filename = metadata["filename"]
        
        # Save to main images directory
        main_path = self.images_dir / filename
        image.convert("RGB").save(main_path, "JPEG", quality=95)
        
        # Save to visibility-based directory
        visibility_dir = self.images_dir / visibility
        visibility_dir.mkdir(exist_ok=True)
        visibility_path = visibility_dir / filename
        image.convert("RGB").save(visibility_path, "JPEG", quality=95)
        
        # Save to type-based directory
        type_dir = self.by_type_dir / img_type
        type_dir.mkdir(exist_ok=True)
        type_path = type_dir / filename
        image.convert("RGB").save(type_path, "JPEG", quality=95)
    
    def _detect_text_in_image(self, image: Image.Image) -> bool:
        """Detect text in image using OCR."""
        if not TESSERACT_AVAILABLE:
            return False
        
        try:
            gray_image = image.convert('L')
            text = pytesseract.image_to_string(gray_image, config='--psm 6')
            meaningful_text = text.strip()
            return len(meaningful_text) > 3 and any(c.isalnum() for c in meaningful_text)
        except Exception as e:
            self.logger.debug(f"Text detection error: {str(e)}")
            return False
    
    def _calculate_image_hash(self, image: Image.Image) -> str:
        """Calculate perceptual hash for duplicate detection."""
        try:
            small_image = image.resize((8, 8), Image.Resampling.LANCZOS)
            gray_image = small_image.convert('L')
            pixels = list(gray_image.getdata())
            avg = sum(pixels) / len(pixels)
            hash_bits = ['1' if pixel > avg else '0' for pixel in pixels]
            hash_string = ''.join(hash_bits)
            return hex(int(hash_string, 2))[2:]
        except Exception as e:
            return hashlib.md5(image.tobytes()).hexdigest()[:16]
    
    def apply_filters_enhanced(self, images: List[Dict[str, Any]], 
                             filter_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply enhanced filtering with visibility and correlation criteria."""
        filtered_images = []
        
        for img in images:
            # Standard filters (size, type, confidence, etc.)
            if not self._passes_standard_filters(img, filter_config):
                continue
            
            # Enhanced filters
            
            # Visibility filter
            if 'visibility_types' in filter_config:
                visibility = img.get('visibility_analysis', {}).get('visibility', 'unknown')
                if visibility not in filter_config['visibility_types']:
                    continue
            
            # Text correlation filter
            if filter_config.get('min_correlation_strength'):
                correlation = img.get('text_correlation', {}).get('correlation_strength', 0)
                if correlation < filter_config['min_correlation_strength']:
                    continue
            
            # Context filter
            if 'required_contexts' in filter_config:
                primary_context = img.get('text_correlation', {}).get('primary_context', 'general')
                if primary_context not in filter_config['required_contexts']:
                    continue
            
            filtered_images.append(img)
        
        return filtered_images
    
    def _passes_standard_filters(self, img: Dict[str, Any], filter_config: Dict[str, Any]) -> bool:
        """Check standard filtering criteria."""
        # Size filters
        if 'min_size' in filter_config:
            min_area = filter_config['min_size'] ** 2
            img_area = img['dimensions']['width'] * img['dimensions']['height']
            if img_area < min_area:
                return False
        
        # Type filters
        if 'exclude_types' in filter_config:
            if img['classification']['type'] in filter_config['exclude_types']:
                return False
        
        # Confidence filter
        if 'confidence_threshold' in filter_config:
            if img['classification']['confidence'] < filter_config['confidence_threshold']:
                return False
        
        # Duplicate filter
        if filter_config.get('no_duplicates', False):
            if img.get('is_duplicate', False):
                return False
        
        return True
    
    def _generate_processing_summary(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive processing summary."""
        if not images:
            return {"total_images": 0}
        
        # Basic counts
        total_images = len(images)
        unique_images = len([img for img in images if not img.get('is_duplicate', False)])
        
        # Classification breakdown
        type_counts = {}
        for img in images:
            img_type = img['classification']['type']
            type_counts[img_type] = type_counts.get(img_type, 0) + 1
        
        # Visibility breakdown
        visibility_counts = {}
        for img in images:
            visibility = img.get('visibility_analysis', {}).get('visibility', 'unknown')
            visibility_counts[visibility] = visibility_counts.get(visibility, 0) + 1
        
        # Correlation analysis
        correlation_scores = [img.get('text_correlation', {}).get('correlation_strength', 0) for img in images]
        avg_correlation = sum(correlation_scores) / len(correlation_scores) if correlation_scores else 0
        
        # Context analysis
        context_counts = {}
        for img in images:
            context = img.get('text_correlation', {}).get('primary_context', 'general')
            context_counts[context] = context_counts.get(context, 0) + 1
        
        return {
            "total_images": total_images,
            "unique_images": unique_images,
            "duplicates": total_images - unique_images,
            "type_distribution": type_counts,
            "visibility_distribution": visibility_counts,
            "average_correlation_strength": round(avg_correlation, 2),
            "context_distribution": context_counts,
            "enhancement_applied": self.enable_enhancement
        }
    
    def save_enhanced_results(self, results: Dict[str, Any], base_filename: str = "enhanced_images") -> Dict[str, str]:
        """Save enhanced extraction results."""
        saved_files = {}
        
        try:
            # Main extraction report
            report_path = self.metadata_dir / f"{base_filename}_extraction_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            saved_files["extraction_report"] = str(report_path)
            
            # Visibility analysis report
            visibility_data = {
                "analysis_date": results.get("extraction_date"),
                "source_pdf": results.get("source_pdf"),
                "visibility_analysis": {
                    img["filename"]: img.get("visibility_analysis", {})
                    for img in results.get("extracted_images", [])
                }
            }
            visibility_path = self.metadata_dir / f"{base_filename}_visibility_analysis.json"
            with open(visibility_path, 'w', encoding='utf-8') as f:
                json.dump(visibility_data, f, indent=2, ensure_ascii=False)
            saved_files["visibility_analysis"] = str(visibility_path)
            
            # Text correlation report
            if any(img.get("text_correlation") for img in results.get("extracted_images", [])):
                correlation_data = {
                    "analysis_date": results.get("extraction_date"),
                    "source_pdf": results.get("source_pdf"),
                    "correlations": [
                        {
                            "filename": img["filename"],
                            "correlation": img.get("text_correlation", {}),
                            "contextual_description": img.get("contextual_description", "")
                        }
                        for img in results.get("extracted_images", [])
                        if img.get("text_correlation")
                    ]
                }
                correlation_path = self.metadata_dir / f"{base_filename}_text_correlations.json"
                with open(correlation_path, 'w', encoding='utf-8') as f:
                    json.dump(correlation_data, f, indent=2, ensure_ascii=False)
                saved_files["text_correlations"] = str(correlation_path)
            
            # Processing summary
            summary_path = self.metadata_dir / f"{base_filename}_processing_summary.json"
            summary_data = {
                "processing_date": results.get("extraction_date"),
                "source_pdf": results.get("source_pdf"),
                "summary": results.get("processing_summary", {})
            }
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            saved_files["processing_summary"] = str(summary_path)
            
            self.logger.info(f"Saved enhanced results to {len(saved_files)} files")
            
        except Exception as e:
            self.logger.error(f"Error saving enhanced results: {str(e)}")
            raise
        
        return saved_files


def main():
    """Enhanced main function with comprehensive CLI."""
    parser = argparse.ArgumentParser(
        description="Enhanced PDF Image Extractor with Advanced Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_image_extractor.py energy_report.pdf --visible-only
  python enhanced_image_extractor.py energy_report.pdf --enhance-for-embeddings
  python enhanced_image_extractor.py energy_report.pdf --text-correlation text_content.txt
  python enhanced_image_extractor.py energy_report.pdf --visibility-types visible embedded
  python enhanced_image_extractor.py energy_report.pdf --min-correlation 0.5
        """
    )
    
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--visible-only", action="store_true", help="Extract only visible images")
    parser.add_argument("--enhance-for-embeddings", action="store_true", help="Optimize for embeddings")
    parser.add_argument("--text-correlation", help="Path to text content file for correlation")
    parser.add_argument("--visibility-types", nargs="+", 
                       choices=["visible", "embedded", "background"],
                       help="Include only these visibility types")
    parser.add_argument("--min-correlation", type=float, 
                       help="Minimum text correlation strength (0.0-1.0)")
    parser.add_argument("--required-contexts", nargs="+",
                       help="Required text contexts for inclusion")
    parser.add_argument("--enhancement-profile", 
                       choices=["embedding_optimized", "ocr_ready", "analysis_ready"],
                       default="embedding_optimized", help="Image enhancement profile")
    parser.add_argument("--no-enhancement", action="store_true", help="Disable image enhancement")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-error output")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Set logging level
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR
    
    # Initialize extractor
    enable_enhancement = not args.no_enhancement
    extractor = EnhancedPDFImageExtractor(
        output_dir=args.output_dir,
        enable_enhancement=enable_enhancement,
        log_level=log_level
    )
    
    try:
        print(f"🔍 Starting enhanced image extraction: {args.pdf_path}")
        
        # Load text content for correlation if provided
        text_content = None
        if args.text_correlation and os.path.exists(args.text_correlation):
            with open(args.text_correlation, 'r', encoding='utf-8') as f:
                text_content = f.read()
            print(f"📝 Loaded text content for correlation: {args.text_correlation}")
        
        # Build filter configuration
        filters = {}
        if args.visible_only:
            filters['visibility_types'] = ['visible']
        elif args.visibility_types:
            filters['visibility_types'] = args.visibility_types
        
        if args.min_correlation:
            filters['min_correlation_strength'] = args.min_correlation
        
        if args.required_contexts:
            filters['required_contexts'] = args.required_contexts
        
        # Extract images with enhanced analysis
        results = extractor.extract_images_enhanced(args.pdf_path, text_content, filters)
        
        # Save results
        pdf_name = pathlib.Path(args.pdf_path).stem
        saved_files = extractor.save_enhanced_results(results, pdf_name)
        
        # Print comprehensive summary
        summary = results.get("processing_summary", {})
        print(f"\n✅ Enhanced image extraction completed!")
        print(f"📄 Processed {results.get('total_pages', 0)} pages")
        print(f"🖼️  Extracted {summary.get('total_images', 0)} images ({summary.get('unique_images', 0)} unique)")
        
        # Visibility breakdown
        if summary.get("visibility_distribution"):
            print(f"👁️  Visibility analysis:")
            for visibility, count in summary["visibility_distribution"].items():
                print(f"   {visibility.title()}: {count} images")
        
        # Classification breakdown
        if summary.get("type_distribution"):
            print(f"🏷️  Classification breakdown:")
            for img_type, count in summary["type_distribution"].items():
                print(f"   {img_type.title()}: {count} images")
        
        # Correlation analysis
        if summary.get("average_correlation_strength", 0) > 0:
            print(f"🔗 Average text correlation: {summary['average_correlation_strength']:.2f}")
        
        # Context analysis
        if summary.get("context_distribution"):
            print(f"📋 Context distribution:")
            for context, count in summary["context_distribution"].items():
                print(f"   {context.replace('_', ' ').title()}: {count} images")
        
        print(f"\n📁 Enhanced output files:")
        for file_type, file_path in saved_files.items():
            print(f"   {file_type.replace('_', ' ').title()}: {file_path}")
        
        print(f"\n📂 Image organization:")
        print(f"   Main images: {extractor.images_dir}")
        print(f"   By visibility: {extractor.images_dir}/visible, /embedded, /background")
        print(f"   By type: {extractor.by_type_dir}")
        if enable_enhancement:
            print(f"   Enhanced: {extractor.enhanced_dir}")
        
        print("\n🎉 Enhanced image extraction completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Extraction cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during extraction: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()