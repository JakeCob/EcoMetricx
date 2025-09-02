#!/usr/bin/env python3
"""
PDF Image Extractor using PyMuPDF

This script extracts all embedded images from PDF documents with intelligent
classification, enhancement, and comprehensive metadata generation.

Features:
- Extract all embedded images from PDFs
- Smart image classification (logo/photo/diagram/chart/unknown)
- Image enhancement pipeline for better quality
- Comprehensive metadata generation
- Duplicate detection using perceptual hashing
- Batch processing support
- Location tracking for text cross-referencing

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
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import math

try:
    import pymupdf as fitz
except ImportError:
    print("Error: PyMuPDF is not installed.")
    print("Please install it using: pip install PyMuPDF")
    sys.exit(1)

try:
    from PIL import Image, ImageEnhance, ImageFilter
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


class ImageClassificationError(Exception):
    """Custom exception for image classification errors."""
    pass


class PDFImageExtractor:
    """
    A comprehensive PDF image extractor with classification and enhancement capabilities.
    
    This class provides methods to extract, classify, enhance, and organize images
    from PDF documents with detailed metadata generation.
    """
    
    def __init__(self, 
                 output_dir: str = "output", 
                 enable_enhancement: bool = True,
                 log_level: int = logging.INFO):
        """
        Initialize the PDF Image Extractor.
        
        Args:
            output_dir (str): Directory to save extracted images and metadata
            enable_enhancement (bool): Whether to enable image enhancement
            log_level (int): Logging level
        """
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_enhancement = enable_enhancement
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.enhanced_dir = self.output_dir / "enhanced"
        self.metadata_dir = self.output_dir / "metadata"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.images_dir, self.enhanced_dir, self.metadata_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.logs_dir / "extraction.log"
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Image hash storage for duplicate detection
        self.image_hashes = {}
        
        # Classification confidence thresholds
        self.classification_thresholds = {
            'logo': 0.6,
            'chart': 0.7,
            'diagram': 0.6,
            'photo': 0.5
        }
    
    def extract_images_from_pdf(self, 
                               pdf_path: str, 
                               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract all images from a PDF file with classification and metadata.
        
        Args:
            pdf_path (str): Path to the PDF file
            filters (Optional[Dict]): Filter configuration for image selection
        
        Returns:
            List[Dict]: List of extracted image metadata
        """
        pdf_path = pathlib.Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Starting image extraction from: {pdf_path}")
        
        try:
            # Open PDF
            pdf_doc = fitz.open(str(pdf_path))
            
            # Check if PDF is encrypted
            if pdf_doc.is_encrypted:
                self.logger.warning("PDF is encrypted. Attempting to open without password...")
                if not pdf_doc.authenticate(""):
                    raise ValueError("PDF is password protected and requires authentication")
            
            extracted_images = []
            total_pages = len(pdf_doc)
            
            self.logger.info(f"Processing {total_pages} pages...")
            
            for page_num in range(total_pages):
                page = pdf_doc[page_num]
                page_images = self._extract_images_from_page(page, page_num, pdf_path.stem)
                
                if page_images:
                    extracted_images.extend(page_images)
                    self.logger.debug(f"Extracted {len(page_images)} images from page {page_num}")
            
            pdf_doc.close()
            
            # Apply filters if specified
            if filters:
                extracted_images = self.apply_filters(extracted_images, filters)
            
            self.logger.info(f"Successfully extracted {len(extracted_images)} images")
            return extracted_images
            
        except Exception as e:
            self.logger.error(f"Error extracting images from PDF: {str(e)}")
            raise
    
    def _extract_images_from_page(self, 
                                 page: fitz.Page, 
                                 page_num: int, 
                                 pdf_name: str) -> List[Dict[str, Any]]:
        """
        Extract images from a single PDF page.
        
        Args:
            page (fitz.Page): PDF page object
            page_num (int): Page number (0-based)
            pdf_name (str): Name of the PDF file (without extension)
        
        Returns:
            List[Dict]: List of image metadata from this page
        """
        page_images = []
        image_list = page.get_images()
        
        if not image_list:
            return page_images
        
        page_rect = page.rect
        
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
                    # Fallback coordinates if not found
                    coordinates = {
                        "x": 0,
                        "y": 0,
                        "width": pix.width,
                        "height": pix.height
                    }
                
                # Page information for classification
                page_info = {
                    "page_width": page_rect.width,
                    "page_height": page_rect.height,
                    "page_number": page_num
                }
                
                # Classify image
                classification = self.classify_image(pil_image, page_info, coordinates)
                
                # Generate filename
                filename = f"{pdf_name}_page{page_num}_img{img_index}_{classification['type']}.jpg"
                
                # Calculate image hash for duplicate detection
                img_hash = self._calculate_image_hash(pil_image)
                
                # Check for duplicates
                is_duplicate = img_hash in self.image_hashes
                if not is_duplicate:
                    self.image_hashes[img_hash] = filename
                
                # Create metadata
                metadata = {
                    "filename": filename,
                    "page_number": page_num,
                    "image_index": img_index,
                    "coordinates": coordinates,
                    "original_format": "PNG",  # PyMuPDF extracts as PNG
                    "size_bytes": len(img_data),
                    "dimensions": {
                        "width": pil_image.width,
                        "height": pil_image.height
                    },
                    "classification": classification,
                    "enhancement_applied": False,
                    "text_detected": self._detect_text_in_image(pil_image) if TESSERACT_AVAILABLE else False,
                    "color_palette": self._analyze_color_palette(pil_image),
                    "hash": img_hash,
                    "is_duplicate": is_duplicate,
                    "aspect_ratio": round(pil_image.width / pil_image.height, 2)
                }
                
                # Save original image
                if not is_duplicate:
                    image_path = self.images_dir / filename
                    pil_image.convert("RGB").save(image_path, "JPEG", quality=95)
                    
                    # Enhance image if enabled
                    if self.enable_enhancement:
                        enhanced_image = self.enhance_image(pil_image)
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
    
    def classify_image(self, 
                      image: Image.Image, 
                      page_info: Dict[str, Any], 
                      coordinates: Dict[str, int]) -> Dict[str, Any]:
        """
        Classify an image using heuristic analysis.
        
        Args:
            image (Image.Image): PIL Image object
            page_info (Dict): Information about the page
            coordinates (Dict): Image position and size on page
        
        Returns:
            Dict: Classification result with type, confidence, and reasoning
        """
        try:
            width, height = image.size
            aspect_ratio = width / height
            page_width = page_info["page_width"]
            page_height = page_info["page_height"]
            
            # Position analysis
            x_pos = coordinates["x"] / page_width if page_width > 0 else 0
            y_pos = coordinates["y"] / page_height if page_height > 0 else 0
            width_ratio = coordinates["width"] / page_width if page_width > 0 else 0
            height_ratio = coordinates["height"] / page_height if page_height > 0 else 0
            
            # Size analysis
            area = width * height
            is_small = area < 10000  # Less than 100x100
            is_large = width_ratio > 0.6 or height_ratio > 0.6
            
            # Color analysis
            colors = image.getcolors(maxcolors=256*256*256)
            unique_colors = len(colors) if colors else 1000
            is_limited_palette = unique_colors < 50
            
            # Text detection
            has_text = self._detect_text_in_image(image) if TESSERACT_AVAILABLE else False
            
            # Classification logic
            scores = {
                'logo': 0.0,
                'chart': 0.0,
                'diagram': 0.0,
                'photo': 0.0
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
            if 0.8 <= aspect_ratio <= 1.2:  # Square-ish
                scores['logo'] += 0.2
            
            # Chart indicators
            if is_large:
                scores['chart'] += 0.3
            if has_text:
                scores['chart'] += 0.4
            if is_limited_palette:
                scores['chart'] += 0.3
            if aspect_ratio > 1.2:  # Wide
                scores['chart'] += 0.2
            if 0.3 < x_pos < 0.7 and 0.2 < y_pos < 0.8:  # Central position
                scores['chart'] += 0.2
            
            # Diagram indicators
            if is_limited_palette:
                scores['diagram'] += 0.4
            if has_text:
                scores['diagram'] += 0.3
            if 0.5 <= aspect_ratio <= 2.0:  # Moderate aspect ratio
                scores['diagram'] += 0.2
            if area > 50000:  # Medium to large size
                scores['diagram'] += 0.3
            
            # Photo indicators
            if unique_colors > 1000:  # Rich color palette
                scores['photo'] += 0.4
            if not has_text:
                scores['photo'] += 0.3
            if area > 100000:  # Large size
                scores['photo'] += 0.3
            if 0.6 <= aspect_ratio <= 1.8:  # Common photo ratios
                scores['photo'] += 0.2
            
            # Determine final classification
            best_type = max(scores, key=scores.get)
            confidence = min(scores[best_type], 1.0)
            
            # Generate reasoning
            reasoning_parts = []
            if is_small:
                reasoning_parts.append("small size")
            if is_large:
                reasoning_parts.append("large size")
            if y_pos < 0.2:
                reasoning_parts.append("header position")
            elif y_pos > 0.8:
                reasoning_parts.append("footer position")
            if is_limited_palette:
                reasoning_parts.append("limited color palette")
            if has_text:
                reasoning_parts.append("contains text")
            
            reasoning = ", ".join(reasoning_parts) if reasoning_parts else "general characteristics"
            
            # Apply minimum confidence threshold
            min_confidence = self.classification_thresholds.get(best_type, 0.5)
            if confidence < min_confidence:
                best_type = "unknown"
                confidence = 0.3
                reasoning = "insufficient confidence in classification"
            
            return {
                "type": best_type,
                "confidence": round(confidence, 2),
                "reasoning": reasoning,
                "scores": {k: round(v, 2) for k, v in scores.items()}
            }
            
        except Exception as e:
            self.logger.warning(f"Error classifying image: {str(e)}")
            return {
                "type": "unknown",
                "confidence": 0.0,
                "reasoning": f"classification error: {str(e)}",
                "scores": {}
            }
    
    def enhance_image(self, 
                     image: Image.Image, 
                     enhancement_level: str = "standard") -> Image.Image:
        """
        Apply enhancement pipeline to improve image quality.
        
        Args:
            image (Image.Image): Input image
            enhancement_level (str): Enhancement level (light, standard, aggressive)
        
        Returns:
            Image.Image: Enhanced image
        """
        try:
            enhanced = image.copy()
            
            # Enhancement parameters based on level
            if enhancement_level == "light":
                sharpness_factor = 1.1
                contrast_factor = 1.05
                brightness_factor = 1.02
            elif enhancement_level == "aggressive":
                sharpness_factor = 1.5
                contrast_factor = 1.3
                brightness_factor = 1.1
            else:  # standard
                sharpness_factor = 1.2
                contrast_factor = 1.15
                brightness_factor = 1.05
            
            # Apply noise reduction (slight blur then sharpen)
            enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness_factor)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast_factor)
            
            # Adjust brightness if needed
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(brightness_factor)
            
            # Color balance (slight saturation boost for photos)
            if enhanced.mode in ('RGB', 'RGBA'):
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(1.1)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Error enhancing image: {str(e)}")
            return image
    
    def _detect_text_in_image(self, image: Image.Image) -> bool:
        """
        Detect if an image contains text using OCR.
        
        Args:
            image (Image.Image): Image to analyze
        
        Returns:
            bool: True if text is detected
        """
        if not TESSERACT_AVAILABLE:
            return False
        
        try:
            # Convert to grayscale for better OCR
            gray_image = image.convert('L')
            
            # Get text from image
            text = pytesseract.image_to_string(gray_image, config='--psm 6')
            
            # Consider text detected if we get meaningful content
            meaningful_text = text.strip()
            return len(meaningful_text) > 3 and any(c.isalnum() for c in meaningful_text)
            
        except Exception as e:
            self.logger.debug(f"Text detection error: {str(e)}")
            return False
    
    def _analyze_color_palette(self, image: Image.Image) -> int:
        """
        Analyze the color palette of an image.
        
        Args:
            image (Image.Image): Image to analyze
        
        Returns:
            int: Number of unique colors (capped at 1000 for performance)
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Get color histogram
            colors = image.getcolors(maxcolors=1000)
            return len(colors) if colors else 1000
            
        except Exception:
            return 100  # Default fallback
    
    def _calculate_image_hash(self, image: Image.Image) -> str:
        """
        Calculate perceptual hash for duplicate detection.
        
        Args:
            image (Image.Image): Image to hash
        
        Returns:
            str: Image hash
        """
        try:
            # Resize to small size for comparison
            small_image = image.resize((8, 8), Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            gray_image = small_image.convert('L')
            
            # Get pixel data
            pixels = list(gray_image.getdata())
            
            # Calculate average
            avg = sum(pixels) / len(pixels)
            
            # Create hash based on pixels above/below average
            hash_bits = ['1' if pixel > avg else '0' for pixel in pixels]
            hash_string = ''.join(hash_bits)
            
            # Convert to hex
            return hex(int(hash_string, 2))[2:]
            
        except Exception as e:
            # Fallback to simple hash
            return hashlib.md5(image.tobytes()).hexdigest()[:16]
    
    def apply_filters(self, 
                     images: List[Dict[str, Any]], 
                     filter_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply filtering criteria to extracted images.
        
        Args:
            images (List[Dict]): List of image metadata
            filter_config (Dict): Filter configuration
        
        Returns:
            List[Dict]: Filtered images
        """
        filtered_images = []
        
        for img in images:
            # Size filters
            if 'min_size' in filter_config:
                min_area = filter_config['min_size'] ** 2
                img_area = img['dimensions']['width'] * img['dimensions']['height']
                if img_area < min_area:
                    continue
            
            if 'max_size' in filter_config:
                max_area = filter_config['max_size'] ** 2
                img_area = img['dimensions']['width'] * img['dimensions']['height']
                if img_area > max_area:
                    continue
            
            # Classification filters
            if 'exclude_types' in filter_config:
                if img['classification']['type'] in filter_config['exclude_types']:
                    continue
            
            if 'include_types' in filter_config:
                if img['classification']['type'] not in filter_config['include_types']:
                    continue
            
            # Confidence threshold
            if 'confidence_threshold' in filter_config:
                if img['classification']['confidence'] < filter_config['confidence_threshold']:
                    continue
            
            # Text detection filter
            if filter_config.get('exclude_text_heavy', False):
                if img.get('text_detected', False):
                    continue
            
            # Duplicate filter
            if filter_config.get('no_duplicates', False):
                if img.get('is_duplicate', False):
                    continue
            
            filtered_images.append(img)
        
        return filtered_images
    
    def generate_metadata(self, 
                         extraction_results: List[Dict[str, Any]], 
                         pdf_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for the extraction results.
        
        Args:
            extraction_results (List[Dict]): Results from image extraction
            pdf_path (str): Path to source PDF
        
        Returns:
            Dict: Comprehensive metadata
        """
        pdf_name = pathlib.Path(pdf_path).name
        
        # Basic statistics
        total_images = len(extraction_results)
        unique_images = len([img for img in extraction_results if not img.get('is_duplicate', False)])
        
        # Classification breakdown
        type_counts = {}
        confidence_sum = {}
        for img in extraction_results:
            img_type = img['classification']['type']
            confidence = img['classification']['confidence']
            
            type_counts[img_type] = type_counts.get(img_type, 0) + 1
            confidence_sum[img_type] = confidence_sum.get(img_type, 0) + confidence
        
        # Average confidence per type
        avg_confidence = {}
        for img_type in type_counts:
            avg_confidence[img_type] = round(confidence_sum[img_type] / type_counts[img_type], 2)
        
        # Page distribution
        page_counts = {}
        for img in extraction_results:
            page = img['page_number']
            page_counts[page] = page_counts.get(page, 0) + 1
        
        metadata = {
            "extraction_date": datetime.now().isoformat(),
            "source_pdf": pdf_name,
            "total_images": total_images,
            "unique_images": unique_images,
            "duplicates_found": total_images - unique_images,
            "classification_summary": {
                "type_counts": type_counts,
                "average_confidence": avg_confidence
            },
            "page_distribution": page_counts,
            "enhancement_applied": self.enable_enhancement,
            "images": extraction_results
        }
        
        return metadata
    
    def save_metadata(self, 
                     metadata: Dict[str, Any], 
                     filename: str = "extraction_report.json") -> str:
        """
        Save metadata to JSON file.
        
        Args:
            metadata (Dict): Metadata to save
            filename (str): Output filename
        
        Returns:
            str: Path to saved file
        """
        output_path = self.metadata_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved metadata to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")
            raise
    
    def detect_duplicates(self, image_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect duplicate images using perceptual hashing.
        
        Args:
            image_list (List[Dict]): List of image metadata
        
        Returns:
            List[Dict]: Updated list with duplicate information
        """
        hash_groups = {}
        
        for img in image_list:
            img_hash = img.get('hash', '')
            if img_hash in hash_groups:
                hash_groups[img_hash].append(img)
            else:
                hash_groups[img_hash] = [img]
        
        # Mark duplicates
        for hash_value, images in hash_groups.items():
            if len(images) > 1:
                # Keep first as original, mark others as duplicates
                for i, img in enumerate(images[1:], 1):
                    img['is_duplicate'] = True
                    img['duplicate_of'] = images[0]['filename']
        
        return image_list


def main():
    """
    Main function with comprehensive CLI interface.
    """
    parser = argparse.ArgumentParser(
        description="Extract images from PDF with intelligent classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_extractor.py document.pdf
  python image_extractor.py document.pdf --output-dir custom_output
  python image_extractor.py document.pdf --min-size 100 --max-size 1000
  python image_extractor.py document.pdf --exclude-text-heavy --confidence-threshold 0.7
  python image_extractor.py document.pdf --enhance --no-duplicates
        """
    )
    
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output-dir", default="output", 
                       help="Output directory (default: output)")
    parser.add_argument("--min-size", type=int, 
                       help="Minimum image size (width or height in pixels)")
    parser.add_argument("--max-size", type=int, 
                       help="Maximum image size (width or height in pixels)")
    parser.add_argument("--exclude-text-heavy", action="store_true",
                       help="Exclude images with significant text content")
    parser.add_argument("--confidence-threshold", type=float, default=0.0,
                       help="Minimum classification confidence (0.0-1.0)")
    parser.add_argument("--enhance", action="store_true",
                       help="Enable image enhancement")
    parser.add_argument("--no-enhancement", action="store_true",
                       help="Disable image enhancement")
    parser.add_argument("--no-duplicates", action="store_true",
                       help="Exclude duplicate images")
    parser.add_argument("--include-types", nargs="+", 
                       choices=["logo", "photo", "diagram", "chart", "unknown"],
                       help="Include only these image types")
    parser.add_argument("--exclude-types", nargs="+", 
                       choices=["logo", "photo", "diagram", "chart", "unknown"],
                       help="Exclude these image types")
    parser.add_argument("--enhancement-level", choices=["light", "standard", "aggressive"],
                       default="standard", help="Enhancement intensity")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress non-error output")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    if args.enhance and args.no_enhancement:
        print("Error: --enhance and --no-enhancement cannot be used together")
        sys.exit(1)
    
    if args.include_types and args.exclude_types:
        if set(args.include_types) & set(args.exclude_types):
            print("Error: include-types and exclude-types cannot overlap")
            sys.exit(1)
    
    # Set log level
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR
    
    # Initialize extractor
    enable_enhancement = not args.no_enhancement
    if args.enhance:
        enable_enhancement = True
    
    extractor = PDFImageExtractor(
        output_dir=args.output_dir,
        enable_enhancement=enable_enhancement,
        log_level=log_level
    )
    
    try:
        print(f"🔍 Extracting images from: {args.pdf_path}")
        
        # Build filter configuration
        filters = {}
        if args.min_size:
            filters['min_size'] = args.min_size
        if args.max_size:
            filters['max_size'] = args.max_size
        if args.exclude_text_heavy:
            filters['exclude_text_heavy'] = True
        if args.confidence_threshold > 0:
            filters['confidence_threshold'] = args.confidence_threshold
        if args.no_duplicates:
            filters['no_duplicates'] = True
        if args.include_types:
            filters['include_types'] = args.include_types
        if args.exclude_types:
            filters['exclude_types'] = args.exclude_types
        
        # Extract images
        extracted_images = extractor.extract_images_from_pdf(args.pdf_path, filters)
        
        if not extracted_images:
            print("⚠️  No images found matching the specified criteria")
            sys.exit(0)
        
        # Generate and save metadata
        metadata = extractor.generate_metadata(extracted_images, args.pdf_path)
        metadata_file = extractor.save_metadata(metadata)
        
        # Save detailed location data
        locations_data = {
            "pdf_name": pathlib.Path(args.pdf_path).name,
            "extraction_date": datetime.now().isoformat(),
            "image_locations": [
                {
                    "filename": img["filename"],
                    "page_number": img["page_number"],
                    "coordinates": img["coordinates"],
                    "classification": img["classification"]["type"]
                }
                for img in extracted_images
            ]
        }
        
        locations_file = extractor.save_metadata(locations_data, "image_locations.json")
        
        # Save classification confidence data
        confidence_data = {
            "extraction_date": datetime.now().isoformat(),
            "confidence_summary": {
                img["filename"]: {
                    "type": img["classification"]["type"],
                    "confidence": img["classification"]["confidence"],
                    "reasoning": img["classification"]["reasoning"]
                }
                for img in extracted_images
            }
        }
        
        confidence_file = extractor.save_metadata(confidence_data, "classification_confidence.json")
        
        # Print summary
        print(f"\n✅ Successfully extracted {len(extracted_images)} images")
        print(f"📁 Images saved to: {extractor.images_dir}")
        if enable_enhancement:
            print(f"🎨 Enhanced images saved to: {extractor.enhanced_dir}")
        print(f"📊 Metadata saved to: {metadata_file}")
        print(f"📍 Locations saved to: {locations_file}")
        print(f"🎯 Classifications saved to: {confidence_file}")
        
        # Classification summary
        type_counts = metadata["classification_summary"]["type_counts"]
        if type_counts:
            print("\n📊 Classification Summary:")
            for img_type, count in type_counts.items():
                avg_conf = metadata["classification_summary"]["average_confidence"][img_type]
                print(f"  {img_type.title()}: {count} images (avg confidence: {avg_conf})")
        
        # Duplicate summary
        if metadata["duplicates_found"] > 0:
            print(f"\n🔄 Found {metadata['duplicates_found']} duplicate images")
        
        print("\n🎉 Image extraction completed successfully!")
        
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
    # Add missing import for io
    import io
    main()