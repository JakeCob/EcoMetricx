#!/usr/bin/env python3
"""
Visual Element Extraction System for EcoMetricx
==============================================

Advanced visual element extraction system that processes PDF screenshots to identify
and extract images, tables, and charts as separate components for multimodal LLM training.

This system provides:
- Layout analysis and region detection
- Table extraction with multiple methods
- Chart/graph data extraction
- Image classification and enhancement
- Spatial relationship mapping
- Integration with existing text extraction

Author: Claude (Anthropic)
Version: 1.0.0
"""

import os
import sys
import json
import logging
import argparse
import csv
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Core dependencies
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Computer vision and image processing
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

from skimage import feature, filters, morphology, segmentation, measure, color
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import closing, remove_small_objects
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

# OCR capabilities
import pytesseract

# Existing integrations
import re
from collections import defaultdict

# Optional scipy for advanced image processing
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ndimage = None


class LayoutAnalyzer:
    """Analyze page layout and identify different visual regions"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.LayoutAnalyzer")
        self.min_region_area = 1000  # Minimum area for a region to be considered
        self.aspect_ratio_thresholds = {
            "very_wide": 4.0,    # Tables, headers
            "wide": 2.0,         # Text blocks
            "square": 0.5,       # Charts, images
            "tall": 0.25         # Sidebars
        }
    
    def analyze_page_layout(self, image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """Detect and classify visual regions on the page"""
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))
                if image is None:
                    raise ValueError(f"Could not load image: {image}")
            
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if OPENCV_AVAILABLE else np.mean(image, axis=2).astype(np.uint8)
            else:
                gray = image
            
            # Get image dimensions
            height, width = gray.shape
            
            # Find regions using multiple methods
            regions_contours = self._detect_regions_by_contours(gray)
            regions_morphology = self._detect_regions_by_morphology(gray)
            regions_edges = self._detect_regions_by_edges(gray)
            
            # Merge and validate regions
            all_regions = regions_contours + regions_morphology + regions_edges
            filtered_regions = self._filter_and_merge_regions(all_regions, (width, height))
            
            # Classify each region
            classified_regions = []
            for i, region in enumerate(filtered_regions):
                region_type = self._classify_region_type(region, gray)
                region_features = self._extract_region_features(region, gray)
                
                classified_region = {
                    "id": i,
                    "type": region_type.get("type", "unknown") if isinstance(region_type, dict) else region_type,
                    "bbox": region["bbox"],
                    "confidence": region_type.get("confidence", 0.5) if isinstance(region_type, dict) else 0.5,
                    "area": region["area"],
                    "aspect_ratio": region["aspect_ratio"],
                    "features": region_features,
                    "position": self._get_position_info(region["bbox"], (width, height)),
                    "classification_details": region_type if isinstance(region_type, dict) else None
                }
                classified_regions.append(classified_region)
            
            # Determine overall layout type
            layout_type = self._determine_layout_type(classified_regions, (width, height))
            
            # Generate reading order
            reading_order = self._generate_reading_order(classified_regions)
            
            layout_result = {
                "image_dimensions": {"width": width, "height": height},
                "regions": classified_regions,
                "layout_type": layout_type,
                "reading_order": reading_order,
                "analysis_metadata": {
                    "total_regions": len(classified_regions),
                    "region_types": self._count_region_types(classified_regions),
                    "analysis_date": datetime.now().isoformat()
                }
            }
            
            self.logger.info(f"Layout analysis complete: {len(classified_regions)} regions identified")
            return layout_result
            
        except Exception as e:
            self.logger.error(f"Error in layout analysis: {str(e)}")
            raise
    
    def _detect_regions_by_contours(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect regions using contour detection"""
        regions = []
        
        if not OPENCV_AVAILABLE:
            return regions
        
        try:
            # Apply threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_region_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    regions.append({
                        "bbox": (x, y, w, h),
                        "area": area,
                        "aspect_ratio": aspect_ratio,
                        "method": "contours",
                        "contour": contour
                    })
            
        except Exception as e:
            self.logger.warning(f"Contour detection failed: {str(e)}")
        
        return regions
    
    def _detect_regions_by_morphology(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect regions using morphological operations"""
        regions = []
        
        try:
            # Use scikit-image for morphological operations
            # Apply threshold
            thresh = threshold_otsu(gray)
            binary = gray > thresh
            
            # Morphological closing to connect nearby elements
            try:
                from skimage.morphology import square
                closed = closing(binary, square(5))
            except (ImportError, AttributeError):
                # Use rectangle footprint for newer versions
                try:
                    from skimage.morphology import rectangle
                    closed = closing(binary, rectangle(5, 5))
                except ImportError:
                    # Fallback for even older versions
                    import numpy as np
                    kernel = np.ones((5, 5), dtype=bool)
                    closed = closing(binary, kernel)
            
            # Remove small objects
            cleaned = remove_small_objects(closed, min_size=self.min_region_area)
            
            # Label connected components
            labeled = label(cleaned)
            
            # Extract region properties
            properties = regionprops(labeled)
            
            for prop in properties:
                if prop.area > self.min_region_area:
                    # Get bounding box
                    min_row, min_col, max_row, max_col = prop.bbox
                    x, y = min_col, min_row
                    w, h = max_col - min_col, max_row - min_row
                    aspect_ratio = w / h if h > 0 else 0
                    
                    regions.append({
                        "bbox": (x, y, w, h),
                        "area": prop.area,
                        "aspect_ratio": aspect_ratio,
                        "method": "morphology",
                        "properties": prop
                    })
            
        except Exception as e:
            self.logger.warning(f"Morphological detection failed: {str(e)}")
        
        return regions
    
    def _detect_regions_by_edges(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect regions using edge detection"""
        regions = []
        
        try:
            # Apply Gaussian filter to reduce noise
            blurred = gaussian(gray, sigma=1.0)
            
            # Detect edges using Canny
            edges = feature.canny(blurred, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
            
            # Fill edge-enclosed regions
            if SCIPY_AVAILABLE:
                filled = ndimage.binary_fill_holes(edges)
            else:
                # Fallback without hole filling
                filled = edges
            
            # Label connected components
            labeled = label(filled)
            
            # Extract region properties
            properties = regionprops(labeled)
            
            for prop in properties:
                if prop.area > self.min_region_area:
                    min_row, min_col, max_row, max_col = prop.bbox
                    x, y = min_col, min_row
                    w, h = max_col - min_col, max_row - min_row
                    aspect_ratio = w / h if h > 0 else 0
                    
                    regions.append({
                        "bbox": (x, y, w, h),
                        "area": prop.area,
                        "aspect_ratio": aspect_ratio,
                        "method": "edges",
                        "properties": prop
                    })
            
        except Exception as e:
            self.logger.warning(f"Edge detection failed: {str(e)}")
        
        return regions
    
    def _filter_and_merge_regions(self, regions: List[Dict[str, Any]], image_dims: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Filter overlapping regions and merge similar ones"""
        if not regions:
            return []
        
        width, height = image_dims
        
        # Filter regions by size and position
        filtered_regions = []
        for region in regions:
            x, y, w, h = region["bbox"]
            
            # Skip regions that are too small or too large
            if w < 50 or h < 50:  # Too small
                continue
            if w > width * 0.95 or h > height * 0.95:  # Too large (likely whole page)
                continue
            
            # Skip regions at the very edges (likely noise)
            if x < 5 or y < 5 or (x + w) > (width - 5) or (y + h) > (height - 5):
                continue
            
            filtered_regions.append(region)
        
        # Remove overlapping regions (keep the one with larger area)
        merged_regions = []
        filtered_regions.sort(key=lambda r: r["area"], reverse=True)
        
        for region in filtered_regions:
            overlaps = False
            for existing in merged_regions:
                if self._regions_overlap(region["bbox"], existing["bbox"]):
                    overlaps = True
                    break
            
            if not overlaps:
                merged_regions.append(region)
        
        return merged_regions
    
    def _regions_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int], threshold: float = 0.3) -> bool:
        """Check if two bounding boxes overlap significantly"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left >= right or top >= bottom:
            return False
        
        intersection_area = (right - left) * (bottom - top)
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Calculate overlap ratio
        overlap_ratio = intersection_area / min(area1, area2)
        return overlap_ratio > threshold
    
    def _classify_region_type(self, region: Dict[str, Any], gray: np.ndarray) -> Dict[str, Any]:
        """Classify what type of visual element this region contains"""
        x, y, w, h = region["bbox"]
        aspect_ratio = region["aspect_ratio"]
        area = region["area"]
        
        # Extract region image
        region_image = gray[y:y+h, x:x+w]
        
        # Initialize classification scores
        scores = {
            "table": 0.0,
            "chart": 0.0,
            "image": 0.0,
            "text": 0.0,
            "header": 0.0,
            "footer": 0.0
        }
        
        # Size-based features
        if aspect_ratio > self.aspect_ratio_thresholds["very_wide"]:
            scores["table"] += 0.3
            scores["header"] += 0.2
        elif aspect_ratio > self.aspect_ratio_thresholds["wide"]:
            scores["text"] += 0.3
            scores["table"] += 0.2
        elif aspect_ratio < self.aspect_ratio_thresholds["square"]:
            scores["chart"] += 0.3
            scores["image"] += 0.2
        
        # Position-based features
        image_height = gray.shape[0]
        if y < image_height * 0.2:  # Top of page
            scores["header"] += 0.3
        elif y > image_height * 0.8:  # Bottom of page
            scores["footer"] += 0.3
        
        # Content-based analysis
        content_scores = self._analyze_region_content(region_image)
        for key, value in content_scores.items():
            scores[key] += value * 0.5
        
        # Determine best classification
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return {
            "type": best_type,
            "confidence": min(confidence, 1.0),
            "scores": scores
        }
    
    def _analyze_region_content(self, region_image: np.ndarray) -> Dict[str, float]:
        """Analyze region content to determine type"""
        scores = defaultdict(float)
        
        if region_image.size == 0:
            return scores
        
        try:
            # Line detection for tables
            if OPENCV_AVAILABLE:
                edges = cv2.Canny(region_image, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=5)
                if lines is not None:
                    horizontal_lines = sum(1 for line in lines if abs(line[0][1] - line[0][3]) < 10)
                    vertical_lines = sum(1 for line in lines if abs(line[0][0] - line[0][2]) < 10)
                    
                    if horizontal_lines > 2 and vertical_lines > 2:
                        scores["table"] += 0.6
                    elif horizontal_lines > 1 or vertical_lines > 1:
                        scores["table"] += 0.3
            
            # Texture analysis for different content types
            # Calculate standard deviation (texture measure)
            texture = np.std(region_image)
            if texture > 50:  # High texture variation
                scores["image"] += 0.3
                scores["chart"] += 0.2
            elif texture < 20:  # Low texture (uniform areas)
                scores["text"] += 0.3
            
            # Edge density analysis
            edges = feature.canny(region_image, sigma=1.0)
            edge_density = np.sum(edges) / edges.size
            
            if edge_density > 0.1:  # High edge density
                scores["chart"] += 0.4
                scores["table"] += 0.2
            elif edge_density < 0.02:  # Low edge density
                scores["text"] += 0.4
            
        except Exception as e:
            self.logger.warning(f"Content analysis failed: {str(e)}")
        
        return dict(scores)
    
    def _extract_region_features(self, region: Dict[str, Any], gray: np.ndarray) -> Dict[str, Any]:
        """Extract detailed features for a region"""
        x, y, w, h = region["bbox"]
        region_image = gray[y:y+h, x:x+w]
        
        features = {
            "dimensions": {"width": w, "height": h},
            "area": region["area"],
            "aspect_ratio": region["aspect_ratio"],
            "position": {"x": x, "y": y}
        }
        
        if region_image.size > 0:
            # Statistical features
            features["intensity_stats"] = {
                "mean": float(np.mean(region_image)),
                "std": float(np.std(region_image)),
                "min": int(np.min(region_image)),
                "max": int(np.max(region_image))
            }
            
            # Texture features
            try:
                # Local Binary Pattern
                lbp = feature.local_binary_pattern(region_image, P=8, R=1, method='uniform')
                features["texture_lbp_variance"] = float(np.var(lbp))
                
                # Gray-Level Co-occurrence Matrix features
                if region_image.shape[0] > 5 and region_image.shape[1] > 5:
                    glcm = feature.graycomatrix(region_image.astype(np.uint8), [1], [0], 256, symmetric=True, normed=True)
                    features["texture_contrast"] = float(feature.graycoprops(glcm, 'contrast')[0, 0])
                    features["texture_dissimilarity"] = float(feature.graycoprops(glcm, 'dissimilarity')[0, 0])
            except Exception as e:
                self.logger.warning(f"Texture analysis failed: {str(e)}")
        
        return features
    
    def _get_position_info(self, bbox: Tuple[int, int, int, int], image_dims: Tuple[int, int]) -> Dict[str, Any]:
        """Get position information for a region"""
        x, y, w, h = bbox
        img_width, img_height = image_dims
        
        # Calculate relative positions
        center_x = x + w // 2
        center_y = y + h // 2
        
        position_info = {
            "absolute": {"x": x, "y": y, "center_x": center_x, "center_y": center_y},
            "relative": {
                "x": x / img_width,
                "y": y / img_height,
                "center_x": center_x / img_width,
                "center_y": center_y / img_height
            },
            "quadrant": self._get_quadrant(center_x, center_y, img_width, img_height),
            "zone": self._get_zone(y, img_height)
        }
        
        return position_info
    
    def _get_quadrant(self, x: int, y: int, width: int, height: int) -> str:
        """Determine which quadrant the region center is in"""
        if x < width // 2:
            return "top_left" if y < height // 2 else "bottom_left"
        else:
            return "top_right" if y < height // 2 else "bottom_right"
    
    def _get_zone(self, y: int, height: int) -> str:
        """Determine which vertical zone the region is in"""
        if y < height * 0.2:
            return "header"
        elif y > height * 0.8:
            return "footer"
        else:
            return "body"
    
    def _determine_layout_type(self, regions: List[Dict[str, Any]], image_dims: Tuple[int, int]) -> str:
        """Determine overall page layout type"""
        if not regions:
            return "empty"
        
        width, height = image_dims
        
        # Count regions in different zones
        left_regions = sum(1 for r in regions if r["position"]["relative"]["center_x"] < 0.4)
        right_regions = sum(1 for r in regions if r["position"]["relative"]["center_x"] > 0.6)
        center_regions = sum(1 for r in regions if 0.4 <= r["position"]["relative"]["center_x"] <= 0.6)
        
        # Determine layout
        if left_regions > 0 and right_regions > 0:
            return "multi_column"
        elif len(regions) == 1:
            return "single_element"
        elif any(r["type"] == "table" for r in regions) and any(r["type"] == "chart" for r in regions):
            return "mixed_data"
        else:
            return "single_column"
    
    def _generate_reading_order(self, regions: List[Dict[str, Any]]) -> List[int]:
        """Generate reading order for regions"""
        # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
        sorted_regions = sorted(regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))
        return [r["id"] for r in sorted_regions]
    
    def _count_region_types(self, regions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count regions by type"""
        type_counts = defaultdict(int)
        for region in regions:
            type_counts[region["type"]] += 1
        return dict(type_counts)


class TableExtractor:
    """Extract and process table data from image regions"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TableExtractor")
        self.ocr_config = "--oem 3 --psm 6"
    
    def extract_tables_from_region(self, region_image: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract table data using multiple methods"""
        try:
            self.logger.info(f"Extracting table from region {region_info.get('id', 'unknown')}")
            
            # Try different table extraction methods
            results = {}
            
            # Method 1: Contour-based detection
            if OPENCV_AVAILABLE:
                contour_result = self._extract_with_contours(region_image)
                if contour_result:
                    results["contour_method"] = contour_result
            
            # Method 2: Line detection approach
            line_result = self._extract_with_line_detection(region_image)
            if line_result:
                results["line_method"] = line_result
            
            # Method 3: Grid-based approach
            grid_result = self._extract_with_grid_analysis(region_image)
            if grid_result:
                results["grid_method"] = grid_result
            
            # Select best result
            best_result = self._select_best_table_result(results)
            
            # Enhance with OCR
            if best_result:
                best_result = self._enhance_with_ocr(region_image, best_result)
            
            return {
                "extraction_successful": best_result is not None,
                "method_used": best_result.get("method", "none") if best_result else "none",
                "table_data": best_result,
                "all_results": results,
                "region_info": region_info
            }
            
        except Exception as e:
            self.logger.error(f"Table extraction failed: {str(e)}")
            return {
                "extraction_successful": False,
                "error": str(e),
                "region_info": region_info
            }
    
    def _extract_with_contours(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract table using contour detection"""
        if not OPENCV_AVAILABLE:
            return None
        
        try:
            # Convert to binary
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter rectangular contours (potential cells)
            cells = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum cell area
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if roughly rectangular (4 corners)
                    if len(approx) >= 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio
                            cells.append({
                                "bbox": (x, y, w, h),
                                "area": area,
                                "contour": contour
                            })
            
            if len(cells) < 4:  # Need at least 4 cells for a table
                return None
            
            # Group cells into rows and columns
            table_structure = self._organize_cells_into_grid(cells)
            
            return {
                "method": "contours",
                "cells": cells,
                "structure": table_structure,
                "confidence": min(len(cells) / 20, 1.0)  # More cells = higher confidence
            }
            
        except Exception as e:
            self.logger.warning(f"Contour-based table extraction failed: {str(e)}")
            return None
    
    def _extract_with_line_detection(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract table using line detection"""
        try:
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if OPENCV_AVAILABLE else np.mean(image, axis=2).astype(np.uint8)
            
            # Detect edges
            edges = feature.canny(gray, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
            
            # Use HoughLines if OpenCV available, otherwise use a simpler approach
            if OPENCV_AVAILABLE:
                lines = cv2.HoughLinesP(edges.astype(np.uint8), 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=5)
                
                if lines is None:
                    return None
                
                # Separate horizontal and vertical lines
                horizontal_lines = []
                vertical_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y1 - y2) < 10:  # Horizontal line
                        horizontal_lines.append((x1, y1, x2, y2))
                    elif abs(x1 - x2) < 10:  # Vertical line
                        vertical_lines.append((x1, y1, x2, y2))
                
                if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
                    return None
                
                # Find intersections to determine cell boundaries
                intersections = self._find_line_intersections(horizontal_lines, vertical_lines)
                
                return {
                    "method": "line_detection",
                    "horizontal_lines": horizontal_lines,
                    "vertical_lines": vertical_lines,
                    "intersections": intersections,
                    "confidence": min((len(horizontal_lines) + len(vertical_lines)) / 20, 1.0)
                }
            else:
                # Simple line detection without OpenCV
                return self._simple_line_detection(gray)
            
        except Exception as e:
            self.logger.warning(f"Line-based table extraction failed: {str(e)}")
            return None
    
    def _extract_with_grid_analysis(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract table using grid pattern analysis"""
        try:
            gray = image if len(image.shape) == 2 else np.mean(image, axis=2).astype(np.uint8)
            height, width = gray.shape
            
            # Apply threshold to get binary image
            thresh = threshold_otsu(gray)
            binary = gray < thresh  # Invert: text is True, background is False
            
            # Analyze row patterns
            row_profiles = np.mean(binary, axis=1)  # Average across width
            col_profiles = np.mean(binary, axis=0)  # Average across height
            
            # Find regular patterns that might indicate table structure
            row_peaks = self._find_peaks_in_profile(row_profiles, height)
            col_peaks = self._find_peaks_in_profile(col_profiles, width)
            
            if len(row_peaks) < 2 or len(col_peaks) < 2:
                return None
            
            # Check if the spacing is regular (indicating a table)
            row_spacing_regular = self._check_regular_spacing(row_peaks)
            col_spacing_regular = self._check_regular_spacing(col_peaks)
            
            if not (row_spacing_regular or col_spacing_regular):
                return None
            
            return {
                "method": "grid_analysis",
                "row_boundaries": row_peaks,
                "col_boundaries": col_peaks,
                "row_profiles": row_profiles.tolist(),
                "col_profiles": col_profiles.tolist(),
                "confidence": 0.7 if (row_spacing_regular and col_spacing_regular) else 0.5
            }
            
        except Exception as e:
            self.logger.warning(f"Grid-based table extraction failed: {str(e)}")
            return None
    
    def _organize_cells_into_grid(self, cells: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Organize detected cells into a grid structure"""
        if not cells:
            return {"rows": 0, "cols": 0, "grid": []}
        
        # Sort cells by position
        cells.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))  # Sort by y, then x
        
        # Group into rows based on y-coordinate
        rows = []
        current_row = []
        current_y = cells[0]["bbox"][1]
        tolerance = 20  # Y-coordinate tolerance for same row
        
        for cell in cells:
            x, y, w, h = cell["bbox"]
            if abs(y - current_y) <= tolerance:
                current_row.append(cell)
            else:
                if current_row:
                    # Sort current row by x-coordinate
                    current_row.sort(key=lambda c: c["bbox"][0])
                    rows.append(current_row)
                current_row = [cell]
                current_y = y
        
        # Add the last row
        if current_row:
            current_row.sort(key=lambda c: c["bbox"][0])
            rows.append(current_row)
        
        # Create grid structure
        grid = []
        max_cols = max(len(row) for row in rows) if rows else 0
        
        for row in rows:
            grid_row = []
            for i in range(max_cols):
                if i < len(row):
                    x, y, w, h = row[i]["bbox"]
                    grid_row.append({
                        "bbox": (x, y, w, h),
                        "area": row[i]["area"]
                    })
                else:
                    grid_row.append(None)  # Missing cell
            grid.append(grid_row)
        
        return {
            "rows": len(grid),
            "cols": max_cols,
            "grid": grid
        }
    
    def _find_line_intersections(self, h_lines: List[Tuple], v_lines: List[Tuple]) -> List[Tuple[int, int]]:
        """Find intersections between horizontal and vertical lines"""
        intersections = []
        
        for h_line in h_lines:
            hx1, hy1, hx2, hy2 = h_line
            for v_line in v_lines:
                vx1, vy1, vx2, vy2 = v_line
                
                # Check if lines intersect
                if (min(hx1, hx2) <= vx1 <= max(hx1, hx2) and
                    min(vy1, vy2) <= hy1 <= max(vy1, vy2)):
                    intersections.append((vx1, hy1))
        
        return intersections
    
    def _simple_line_detection(self, gray: np.ndarray) -> Optional[Dict[str, Any]]:
        """Simple line detection without OpenCV"""
        try:
            # Use edge detection from scikit-image
            edges = feature.canny(gray, sigma=1.0)
            
            # Find potential horizontal lines by checking row sums
            horizontal_candidates = []
            for y in range(edges.shape[0]):
                row_sum = np.sum(edges[y, :])
                if row_sum > edges.shape[1] * 0.3:  # At least 30% of width has edges
                    horizontal_candidates.append(y)
            
            # Find potential vertical lines by checking column sums
            vertical_candidates = []
            for x in range(edges.shape[1]):
                col_sum = np.sum(edges[:, x])
                if col_sum > edges.shape[0] * 0.3:  # At least 30% of height has edges
                    vertical_candidates.append(x)
            
            if len(horizontal_candidates) < 2 or len(vertical_candidates) < 2:
                return None
            
            return {
                "method": "simple_line_detection",
                "horizontal_candidates": horizontal_candidates,
                "vertical_candidates": vertical_candidates,
                "confidence": 0.6
            }
            
        except Exception:
            return None
    
    def _find_peaks_in_profile(self, profile: np.ndarray, dimension: int) -> List[int]:
        """Find peaks in intensity profile that might indicate table boundaries"""
        # Simple peak detection
        peaks = []
        threshold = np.mean(profile) + np.std(profile)
        
        for i in range(1, len(profile) - 1):
            if profile[i] > threshold and profile[i] > profile[i-1] and profile[i] > profile[i+1]:
                peaks.append(i)
        
        return peaks
    
    def _check_regular_spacing(self, positions: List[int]) -> bool:
        """Check if positions have regular spacing (indicating table structure)"""
        if len(positions) < 3:
            return False
        
        # Calculate spacings
        spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        
        # Check if spacings are reasonably consistent
        mean_spacing = np.mean(spacings)
        std_spacing = np.std(spacings)
        
        # Consider regular if standard deviation is less than 30% of mean
        return std_spacing < 0.3 * mean_spacing
    
    def _select_best_table_result(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select the best table extraction result"""
        if not results:
            return None
        
        # Score each result
        best_result = None
        best_score = 0
        
        for method, result in results.items():
            confidence = result.get("confidence", 0)
            
            # Additional scoring based on method reliability
            if method == "contours" and result.get("cells"):
                score = confidence * len(result["cells"]) * 0.1
            elif method == "line_detection" and result.get("intersections"):
                score = confidence * len(result["intersections"]) * 0.05
            elif method == "grid_analysis":
                score = confidence
            else:
                score = confidence * 0.5
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result
    
    def _enhance_with_ocr(self, region_image: np.ndarray, table_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance table result with OCR text extraction"""
        try:
            # Convert to PIL Image for OCR
            if len(region_image.shape) == 3:
                pil_image = Image.fromarray(region_image)
            else:
                pil_image = Image.fromarray(region_image).convert('RGB')
            
            # Extract text from entire region
            full_text = pytesseract.image_to_string(pil_image, config=self.ocr_config)
            
            # Try to parse as table
            text_lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            
            # Simple table parsing - split by whitespace
            table_data = []
            for line in text_lines:
                # Split by multiple spaces or tabs
                cells = re.split(r'\s{2,}|\t', line)
                if len(cells) > 1:  # Only include lines with multiple columns
                    table_data.append(cells)
            
            table_result["ocr_enhancement"] = {
                "full_text": full_text,
                "parsed_table": table_data,
                "text_lines": text_lines
            }
            
            return table_result
            
        except Exception as e:
            self.logger.warning(f"OCR enhancement failed: {str(e)}")
            return table_result


class ChartExtractor:
    """Extract and analyze chart/graph data from image regions"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ChartExtractor")
        self.chart_types = {
            "bar_chart": {"min_bars": 2, "aspect_ratio_range": (0.5, 3.0)},
            "line_chart": {"min_points": 3, "aspect_ratio_range": (1.0, 3.0)},
            "pie_chart": {"circularity_threshold": 0.7, "aspect_ratio_range": (0.8, 1.2)},
            "scatter_plot": {"min_points": 5, "aspect_ratio_range": (0.8, 2.0)}
        }
    
    def extract_charts_from_region(self, region_image: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-method chart extraction"""
        try:
            self.logger.info(f"Extracting chart from region {region_info.get('id', 'unknown')}")
            
            # 1. Chart type classification
            chart_classification = self._classify_chart_type(region_image)
            
            # 2. Extract common chart elements
            chart_elements = self._extract_chart_elements(region_image)
            
            # 3. Specialized extraction based on type
            chart_data = None
            chart_type = chart_classification.get("type", "unknown")
            
            if chart_type == "bar_chart":
                chart_data = self._extract_bar_chart(region_image)
            elif chart_type == "line_chart":
                chart_data = self._extract_line_chart(region_image)
            elif chart_type == "pie_chart":
                chart_data = self._extract_pie_chart(region_image)
            else:
                chart_data = self._extract_generic_chart(region_image)
            
            # 4. Combine results
            extraction_result = {
                "extraction_successful": chart_data is not None,
                "chart_type": chart_classification,
                "chart_elements": chart_elements,
                "chart_data": chart_data,
                "region_info": region_info
            }
            
            return extraction_result
            
        except Exception as e:
            self.logger.error(f"Chart extraction failed: {str(e)}")
            return {
                "extraction_successful": False,
                "error": str(e),
                "region_info": region_info
            }
    
    def _classify_chart_type(self, image: np.ndarray) -> Dict[str, Any]:
        """Identify type of chart/graph"""
        try:
            gray = image if len(image.shape) == 2 else np.mean(image, axis=2).astype(np.uint8)
            height, width = gray.shape
            aspect_ratio = width / height
            
            # Initialize classification scores
            type_scores = {chart_type: 0.0 for chart_type in self.chart_types.keys()}
            
            # Aspect ratio analysis
            for chart_type, config in self.chart_types.items():
                ar_range = config["aspect_ratio_range"]
                if ar_range[0] <= aspect_ratio <= ar_range[1]:
                    type_scores[chart_type] += 0.3
            
            # Shape analysis
            edges = feature.canny(gray, sigma=1.0)
            
            # Look for circular shapes (pie charts)
            try:
                # Use Hough circle detection if available
                if OPENCV_AVAILABLE:
                    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, min(height, width)//4,
                                             param1=50, param2=30, minRadius=min(height, width)//10,
                                             maxRadius=min(height, width)//2)
                    if circles is not None:
                        type_scores["pie_chart"] += 0.6
            except Exception:
                pass
            
            # Look for rectangular/bar patterns
            if OPENCV_AVAILABLE:
                contours, _ = cv2.findContours((edges * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rectangular_count = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:  # Minimum area
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        if len(approx) == 4:  # Rectangular shape
                            rectangular_count += 1
                
                if rectangular_count >= 2:
                    type_scores["bar_chart"] += 0.5
            
            # Look for line patterns
            if OPENCV_AVAILABLE:
                lines = cv2.HoughLinesP(edges.astype(np.uint8), 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=5)
                if lines is not None:
                    diagonal_lines = 0
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        if abs(x1 - x2) > 10 and abs(y1 - y2) > 10:  # Diagonal line
                            diagonal_lines += 1
                    
                    if diagonal_lines >= 1:
                        type_scores["line_chart"] += 0.4
                        type_scores["scatter_plot"] += 0.3
            
            # Determine best classification
            best_type = max(type_scores, key=type_scores.get)
            confidence = type_scores[best_type]
            
            return {
                "type": best_type,
                "confidence": confidence,
                "scores": type_scores,
                "aspect_ratio": aspect_ratio
            }
            
        except Exception as e:
            self.logger.warning(f"Chart classification failed: {str(e)}")
            return {"type": "unknown", "confidence": 0.0, "error": str(e)}
    
    def _extract_chart_elements(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract common chart components"""
        try:
            # Convert to PIL Image for OCR
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image).convert('RGB')
            
            # Extract all text from chart
            full_text = pytesseract.image_to_string(pil_image, config="--oem 3 --psm 6")
            text_lines = [line.strip() for line in full_text.split('\n') if line.strip()]
            
            elements = {
                "title": self._extract_chart_title(text_lines, image),
                "axis_labels": self._extract_axis_labels(text_lines, image),
                "legend": self._extract_legend(text_lines, image),
                "data_labels": self._extract_data_labels(text_lines, image),
                "full_text": full_text,
                "text_lines": text_lines
            }
            
            return elements
            
        except Exception as e:
            self.logger.warning(f"Chart element extraction failed: {str(e)}")
            return {"error": str(e)}
    
    def _extract_bar_chart(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract data from bar charts"""
        try:
            if not OPENCV_AVAILABLE:
                return self._extract_generic_chart(image)
            
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to isolate bars
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours (potential bars)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter for bar-like rectangles
            bars = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum bar area
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Check if it's bar-like (tall or wide rectangle)
                    if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                        bars.append({
                            "bbox": (x, y, w, h),
                            "area": area,
                            "aspect_ratio": aspect_ratio,
                            "value": h if aspect_ratio < 0.5 else w  # Height for vertical, width for horizontal
                        })
            
            if not bars:
                return None
            
            # Sort bars by position
            bars.sort(key=lambda b: b["bbox"][0])  # Sort by x-coordinate
            
            return {
                "chart_type": "bar_chart",
                "bars": bars,
                "bar_count": len(bars),
                "confidence": min(len(bars) / 10, 1.0)  # More bars = higher confidence
            }
            
        except Exception as e:
            self.logger.warning(f"Bar chart extraction failed: {str(e)}")
            return None
    
    def _extract_line_chart(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract data from line charts"""
        try:
            gray = image if len(image.shape) == 2 else np.mean(image, axis=2).astype(np.uint8)
            
            # Detect edges
            edges = feature.canny(gray, sigma=1.0)
            
            # Find potential data points (edge intersections or high-intensity points)
            # This is a simplified approach - in practice, you'd need more sophisticated methods
            
            # Find peaks in the image that might represent data points
            from scipy import ndimage
            
            # Apply Gaussian filter
            smoothed = ndimage.gaussian_filter(gray, sigma=2)
            
            # Find local maxima
            from skimage.feature import peak_local_maxima
            peaks = peak_local_maxima(smoothed, min_distance=10, threshold_abs=0.3*smoothed.max())
            
            data_points = [(int(p[1]), int(p[0])) for p in peaks]  # Convert (row, col) to (x, y)
            
            if len(data_points) < 3:
                return None
            
            # Sort points by x-coordinate
            data_points.sort(key=lambda p: p[0])
            
            return {
                "chart_type": "line_chart",
                "data_points": data_points,
                "point_count": len(data_points),
                "confidence": min(len(data_points) / 20, 1.0)
            }
            
        except Exception as e:
            self.logger.warning(f"Line chart extraction failed: {str(e)}")
            return None
    
    def _extract_pie_chart(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract data from pie charts"""
        try:
            if not OPENCV_AVAILABLE:
                return self._extract_generic_chart(image)
            
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect circles
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0]//4,
                                     param1=50, param2=30, 
                                     minRadius=gray.shape[0]//10, 
                                     maxRadius=gray.shape[0]//2)
            
            if circles is None:
                return None
            
            circles = np.round(circles[0, :]).astype("int")
            
            # For each detected circle, try to identify pie segments
            pie_data = []
            for (x, y, r) in circles:
                # Create a mask for the circle
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                
                # Apply mask to original image
                masked = cv2.bitwise_and(gray, gray, mask=mask)
                
                # Try to detect pie segments using contour analysis
                _, binary = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                segments = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > r * r * 0.1:  # At least 10% of circle area
                        segments.append({
                            "area": area,
                            "contour": contour,
                            "percentage": (area / (np.pi * r * r)) * 100
                        })
                
                pie_data.append({
                    "center": (x, y),
                    "radius": r,
                    "segments": segments
                })
            
            if not pie_data:
                return None
            
            return {
                "chart_type": "pie_chart",
                "circles": pie_data,
                "confidence": 0.8 if len(pie_data) == 1 else 0.6
            }
            
        except Exception as e:
            self.logger.warning(f"Pie chart extraction failed: {str(e)}")
            return None
    
    def _extract_generic_chart(self, image: np.ndarray) -> Dict[str, Any]:
        """Generic chart analysis when specific type detection fails"""
        try:
            gray = image if len(image.shape) == 2 else np.mean(image, axis=2).astype(np.uint8)
            height, width = gray.shape
            
            # Basic statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Edge analysis
            edges = feature.canny(gray, sigma=1.0)
            edge_density = np.sum(edges) / edges.size
            
            # Texture analysis
            texture_variance = np.var(gray)
            
            return {
                "chart_type": "generic",
                "statistics": {
                    "mean_intensity": float(mean_intensity),
                    "std_intensity": float(std_intensity),
                    "edge_density": float(edge_density),
                    "texture_variance": float(texture_variance)
                },
                "dimensions": {"width": width, "height": height},
                "confidence": 0.3
            }
            
        except Exception as e:
            self.logger.warning(f"Generic chart extraction failed: {str(e)}")
            return {"chart_type": "unknown", "error": str(e), "confidence": 0.0}
    
    def _extract_chart_title(self, text_lines: List[str], image: np.ndarray) -> Optional[str]:
        """Extract chart title"""
        if not text_lines:
            return None
        
        # Usually the title is at the top and might be the first or one of the first lines
        # Look for lines that are likely titles (longer, at the top)
        height = image.shape[0]
        
        # Simple heuristic: first non-empty line is likely the title
        for line in text_lines:
            if len(line.strip()) > 5:  # Reasonable title length
                return line.strip()
        
        return None
    
    def _extract_axis_labels(self, text_lines: List[str], image: np.ndarray) -> Dict[str, List[str]]:
        """Extract axis labels"""
        # This is a simplified approach - in practice, you'd analyze text position
        labels = {"x_axis": [], "y_axis": []}
        
        # Look for numeric patterns (likely axis values)
        numeric_pattern = re.compile(r'^-?\d+\.?\d*$')
        
        for line in text_lines:
            if numeric_pattern.match(line.strip()):
                # Could be either axis - would need position analysis to determine
                labels["x_axis"].append(line.strip())
        
        return labels
    
    def _extract_legend(self, text_lines: List[str], image: np.ndarray) -> List[str]:
        """Extract legend items"""
        # Look for short text lines that might be legend items
        legend_items = []
        
        for line in text_lines:
            stripped = line.strip()
            if 3 <= len(stripped) <= 30:  # Reasonable legend item length
                legend_items.append(stripped)
        
        return legend_items
    
    def _extract_data_labels(self, text_lines: List[str], image: np.ndarray) -> List[str]:
        """Extract data labels from chart"""
        data_labels = []
        
        # Look for patterns that might be data labels
        for line in text_lines:
            stripped = line.strip()
            # Check for percentage patterns or numeric values
            if re.search(r'\d+%|\d+\.\d+', stripped):
                data_labels.append(stripped)
        
        return data_labels


class VisualImageExtractor:
    """Enhanced image extraction and classification for non-data images"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VisualImageExtractor")
        self.image_types = {
            "logo": {"size_range": (20, 200), "aspect_range": (0.3, 4.0)},
            "photo": {"size_range": (100, 1000), "aspect_range": (0.5, 2.0)},
            "illustration": {"size_range": (50, 500), "aspect_range": (0.4, 2.5)},
            "diagram": {"size_range": (80, 600), "aspect_range": (0.6, 3.0)},
            "icon": {"size_range": (16, 100), "aspect_range": (0.8, 1.2)}
        }
    
    def extract_images_from_region(self, region_image: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process image regions"""
        try:
            self.logger.info(f"Processing image region {region_info.get('id', 'unknown')}")
            
            # 1. Enhance image quality
            enhanced_image = self._enhance_image(region_image)
            
            # 2. Classify image type
            image_classification = self._classify_image_type(enhanced_image, region_info)
            
            # 3. Extract metadata
            metadata = self._extract_image_metadata(enhanced_image, region_info)
            
            # 4. Generate description
            description = self._generate_description(enhanced_image, image_classification)
            
            # 5. Perform OCR if text is expected
            ocr_result = None
            if image_classification.get("likely_contains_text", False):
                ocr_result = self._extract_text_from_image(enhanced_image)
            
            return {
                "extraction_successful": True,
                "image_type": image_classification,
                "enhanced_image": enhanced_image,
                "metadata": metadata,
                "description": description,
                "ocr_result": ocr_result,
                "region_info": region_info
            }
            
        except Exception as e:
            self.logger.error(f"Image extraction failed: {str(e)}")
            return {
                "extraction_successful": False,
                "error": str(e),
                "region_info": region_info
            }
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality"""
        try:
            # Convert to PIL for enhancement
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image).convert('RGB')
            
            # Apply enhancements
            # Increase contrast slightly
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(1.1)
            
            # Slight sharpening
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            # Convert back to numpy array
            return np.array(enhanced)
            
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {str(e)}")
            return image
    
    def _classify_image_type(self, image: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Classify image type"""
        try:
            height, width = image.shape[:2]
            area = width * height
            aspect_ratio = width / height
            
            # Initialize classification scores
            type_scores = {img_type: 0.0 for img_type in self.image_types.keys()}
            
            # Size-based scoring
            for img_type, config in self.image_types.items():
                size_range = config["size_range"]
                aspect_range = config["aspect_range"]
                
                # Size scoring
                min_dim = min(width, height)
                if size_range[0] <= min_dim <= size_range[1]:
                    type_scores[img_type] += 0.3
                
                # Aspect ratio scoring
                if aspect_range[0] <= aspect_ratio <= aspect_range[1]:
                    type_scores[img_type] += 0.3
            
            # Position-based scoring
            position = region_info.get("position", {})
            zone = position.get("zone", "body")
            
            if zone == "header":
                type_scores["logo"] += 0.4
            elif zone == "footer":
                type_scores["logo"] += 0.2
                type_scores["icon"] += 0.2
            
            # Content-based analysis
            content_scores = self._analyze_image_content(image)
            for img_type, score in content_scores.items():
                if img_type in type_scores:
                    type_scores[img_type] += score * 0.4
            
            # Determine best classification
            best_type = max(type_scores, key=type_scores.get)
            confidence = type_scores[best_type]
            
            # Check if likely to contain text
            likely_contains_text = best_type in ["logo", "diagram"] or confidence < 0.5
            
            return {
                "type": best_type,
                "confidence": min(confidence, 1.0),
                "scores": type_scores,
                "likely_contains_text": likely_contains_text,
                "dimensions": {"width": width, "height": height},
                "aspect_ratio": aspect_ratio
            }
            
        except Exception as e:
            self.logger.warning(f"Image classification failed: {str(e)}")
            return {"type": "unknown", "confidence": 0.0, "error": str(e)}
    
    def _analyze_image_content(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze image content for classification"""
        scores = defaultdict(float)
        
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2).astype(np.uint8)
            else:
                gray = image
            
            # Color analysis (if color image)
            if len(image.shape) == 3:
                # Calculate color variance
                color_std = np.std(image, axis=(0, 1))
                color_variance = np.mean(color_std)
                
                if color_variance > 30:  # High color variation
                    scores["photo"] += 0.5
                    scores["illustration"] += 0.3
                elif color_variance < 10:  # Low color variation
                    scores["logo"] += 0.4
                    scores["icon"] += 0.3
            
            # Texture analysis
            texture_variance = np.var(gray)
            if texture_variance > 1000:  # High texture
                scores["photo"] += 0.4
                scores["illustration"] += 0.2
            elif texture_variance < 200:  # Low texture (uniform)
                scores["logo"] += 0.3
                scores["icon"] += 0.3
            
            # Edge analysis
            edges = feature.canny(gray, sigma=1.0)
            edge_density = np.sum(edges) / edges.size
            
            if edge_density > 0.1:  # High edge density
                scores["diagram"] += 0.4
                scores["logo"] += 0.2
            elif edge_density < 0.02:  # Low edge density
                scores["photo"] += 0.3
            
            # Symmetry analysis (simple)
            # Check horizontal symmetry
            left_half = gray[:, :gray.shape[1]//2]
            right_half = np.fliplr(gray[:, gray.shape[1]//2:])
            
            if left_half.shape == right_half.shape:
                symmetry_score = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
                if not np.isnan(symmetry_score) and symmetry_score > 0.7:
                    scores["logo"] += 0.3
                    scores["icon"] += 0.2
            
        except Exception as e:
            self.logger.warning(f"Image content analysis failed: {str(e)}")
        
        return dict(scores)
    
    def _extract_image_metadata(self, image: np.ndarray, region_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from image"""
        try:
            height, width = image.shape[:2]
            
            metadata = {
                "dimensions": {"width": width, "height": height},
                "area": width * height,
                "aspect_ratio": width / height,
                "region_info": region_info
            }
            
            # Color statistics
            if len(image.shape) == 3:
                metadata["color_stats"] = {
                    "mean_rgb": np.mean(image, axis=(0, 1)).tolist(),
                    "std_rgb": np.std(image, axis=(0, 1)).tolist()
                }
            else:
                metadata["intensity_stats"] = {
                    "mean": float(np.mean(image)),
                    "std": float(np.std(image)),
                    "min": int(np.min(image)),
                    "max": int(np.max(image))
                }
            
            # Dominant colors (simplified)
            if len(image.shape) == 3:
                # Reshape image to list of pixels
                pixels = image.reshape(-1, 3)
                # Find unique colors and their counts (simplified approach)
                unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
                dominant_idx = np.argmax(counts)
                metadata["dominant_color"] = unique_colors[dominant_idx].tolist()
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Metadata extraction failed: {str(e)}")
            return {"error": str(e)}
    
    def _generate_description(self, image: np.ndarray, classification: Dict[str, Any]) -> str:
        """Generate descriptive text for image content"""
        try:
            img_type = classification.get("type", "unknown")
            confidence = classification.get("confidence", 0.0)
            dimensions = classification.get("dimensions", {})
            
            # Create basic description
            description_parts = []
            
            # Type and confidence
            if confidence > 0.7:
                description_parts.append(f"This appears to be a {img_type}")
            elif confidence > 0.4:
                description_parts.append(f"This is likely a {img_type}")
            else:
                description_parts.append(f"This may be a {img_type} or other visual element")
            
            # Size description
            width = dimensions.get("width", 0)
            height = dimensions.get("height", 0)
            
            if width > 400 or height > 400:
                description_parts.append("of large size")
            elif width < 100 or height < 100:
                description_parts.append("of small size")
            else:
                description_parts.append("of medium size")
            
            # Aspect ratio description
            aspect_ratio = classification.get("aspect_ratio", 1.0)
            if aspect_ratio > 2.0:
                description_parts.append("with a wide aspect ratio")
            elif aspect_ratio < 0.5:
                description_parts.append("with a tall aspect ratio")
            
            return ". ".join(description_parts) + "."
            
        except Exception as e:
            self.logger.warning(f"Description generation failed: {str(e)}")
            return f"Visual element of type {classification.get('type', 'unknown')}"
    
    def _extract_text_from_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text from image using OCR"""
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image).convert('RGB')
            
            # Extract text
            text = pytesseract.image_to_string(pil_image, config="--oem 3 --psm 6")
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            # Calculate confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                "text": text.strip(),
                "confidence": avg_confidence,
                "word_count": len(text.split()) if text.strip() else 0,
                "has_text": bool(text.strip())
            }
            
        except Exception as e:
            self.logger.warning(f"Image OCR failed: {str(e)}")
            return {"text": "", "confidence": 0, "word_count": 0, "has_text": False, "error": str(e)}


class IntegratedVisualProcessor:
    """Combine all extraction results with spatial relationships"""
    
    def __init__(self, output_dir: str = "output/visual_element_extraction"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["layout_analysis", "tables", "charts", "images", "integrated"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
        
        # Create subdirectories for organized output
        (self.output_dir / "tables" / "extracted").mkdir(exist_ok=True)
        (self.output_dir / "tables" / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "charts" / "extracted").mkdir(exist_ok=True) 
        (self.output_dir / "charts" / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "images" / "metadata").mkdir(exist_ok=True)
        
        # Initialize extractors
        self.layout_analyzer = LayoutAnalyzer()
        self.table_extractor = TableExtractor()
        self.chart_extractor = ChartExtractor()
        self.image_extractor = VisualImageExtractor()
        
        self.logger = logging.getLogger(f"{__name__}.IntegratedVisualProcessor")
    
    def process_pdf_page_visual(self, screenshot_path: Union[str, Path], ocr_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Complete visual processing pipeline"""
        try:
            screenshot_path = Path(screenshot_path)
            base_name = screenshot_path.stem
            
            self.logger.info(f"Processing visual elements from {screenshot_path}")
            
            # 1. Load and analyze layout
            self.logger.info("Analyzing page layout...")
            layout_result = self.layout_analyzer.analyze_page_layout(str(screenshot_path))
            
            # Save layout analysis
            layout_path = self.output_dir / "layout_analysis" / f"{base_name}_layout.json"
            with open(layout_path, 'w', encoding='utf-8') as f:
                json.dump(layout_result, f, indent=2, ensure_ascii=False)
            
            # 2. Load original image for region extraction
            if OPENCV_AVAILABLE:
                image = cv2.imread(str(screenshot_path))
            else:
                from PIL import Image as PILImage
                pil_image = PILImage.open(screenshot_path)
                image = np.array(pil_image)
            
            # 3. Extract each region type
            results = {
                "source_image": str(screenshot_path),
                "processing_date": datetime.now().isoformat(),
                "layout_analysis": layout_result,
                "tables": [],
                "charts": [],
                "images": [],
                "extraction_summary": {}
            }
            
            total_regions = len(layout_result.get("regions", []))
            processed_regions = 0
            
            for region in layout_result.get("regions", []):
                try:
                    # Extract region from image
                    bbox = region["bbox"]
                    x, y, w, h = bbox
                    region_image = image[y:y+h, x:x+w]
                    
                    region_type = region["type"]
                    self.logger.info(f"Processing {region_type} region {region['id']}")
                    
                    if region_type == "table":
                        table_result = self.table_extractor.extract_tables_from_region(region_image, region)
                        if table_result.get("extraction_successful", False):
                            results["tables"].append(table_result)
                            self._save_table_result(table_result, base_name, region["id"], region_image)
                    
                    elif region_type == "chart":
                        chart_result = self.chart_extractor.extract_charts_from_region(region_image, region)
                        if chart_result.get("extraction_successful", False):
                            results["charts"].append(chart_result)
                            self._save_chart_result(chart_result, base_name, region["id"], region_image)
                    
                    elif region_type in ["image", "logo", "photo", "illustration"]:
                        image_result = self.image_extractor.extract_images_from_region(region_image, region)
                        if image_result.get("extraction_successful", False):
                            results["images"].append(image_result)
                            self._save_image_result(image_result, base_name, region["id"], region_image)
                    
                    processed_regions += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process region {region['id']}: {str(e)}")
            
            # 4. Create spatial relationships
            self.logger.info("Creating spatial relationships...")
            spatial_map = self._create_spatial_relationships(layout_result, ocr_results)
            results["spatial_relationships"] = spatial_map
            
            # 5. Generate extraction summary
            results["extraction_summary"] = {
                "total_regions": total_regions,
                "processed_regions": processed_regions,
                "tables_extracted": len(results["tables"]),
                "charts_extracted": len(results["charts"]),
                "images_extracted": len(results["images"]),
                "processing_success_rate": processed_regions / total_regions if total_regions > 0 else 0
            }
            
            # 6. Save integrated results
            integrated_path = self.output_dir / "integrated" / f"{base_name}_visual_extraction.json"
            with open(integrated_path, 'w', encoding='utf-8') as f:
                # Remove numpy arrays and other non-serializable objects for JSON
                json_safe_results = self._make_json_safe(results)
                json.dump(json_safe_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Visual processing completed: {processed_regions}/{total_regions} regions processed")
            return results
            
        except Exception as e:
            self.logger.error(f"Visual processing failed: {str(e)}")
            raise
    
    def _save_table_result(self, table_result: Dict[str, Any], base_name: str, region_id: int, region_image: np.ndarray):
        """Save table extraction result and cropped image"""
        try:
            table_dir = self.output_dir / "tables"
            (table_dir / "extracted").mkdir(exist_ok=True)
            (table_dir / "metadata").mkdir(exist_ok=True)
            
            # Save JSON metadata
            json_path = table_dir / "metadata" / f"{base_name}_region{region_id}_table.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json_safe_result = self._make_json_safe(table_result)
                json.dump(json_safe_result, f, indent=2, ensure_ascii=False)
            
            # Save cropped table image
            image_path = table_dir / "extracted" / f"{base_name}_region{region_id}_table.png"
            if len(region_image.shape) == 3:
                # Convert BGR to RGB if using OpenCV
                if OPENCV_AVAILABLE:
                    region_image = cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(region_image)
            else:
                pil_image = Image.fromarray(region_image).convert('RGB')
            
            pil_image.save(image_path)
            self.logger.info(f"Saved table image: {image_path}")
            
            # Save CSV if table data was parsed
            table_data = table_result.get("table_data", {})
            if table_data and "ocr_enhancement" in table_data:
                parsed_table = table_data["ocr_enhancement"].get("parsed_table", [])
                if parsed_table:
                    csv_path = table_dir / "extracted" / f"{base_name}_region{region_id}_table.csv"
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerows(parsed_table)
                    self.logger.info(f"Saved table CSV: {csv_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save table result: {str(e)}")
    
    def _save_chart_result(self, chart_result: Dict[str, Any], base_name: str, region_id: int, region_image: np.ndarray):
        """Save chart extraction result and cropped image"""
        try:
            chart_dir = self.output_dir / "charts"
            (chart_dir / "extracted").mkdir(exist_ok=True)
            (chart_dir / "metadata").mkdir(exist_ok=True)
            
            # Save JSON metadata
            json_path = chart_dir / "metadata" / f"{base_name}_region{region_id}_chart.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json_safe_result = self._make_json_safe(chart_result)
                json.dump(json_safe_result, f, indent=2, ensure_ascii=False)
            
            # Save cropped chart image
            chart_type = chart_result.get("chart_type", {}).get("type", "chart")
            image_path = chart_dir / "extracted" / f"{base_name}_region{region_id}_{chart_type}.png"
            
            if len(region_image.shape) == 3:
                # Convert BGR to RGB if using OpenCV
                if OPENCV_AVAILABLE:
                    region_image = cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(region_image)
            else:
                pil_image = Image.fromarray(region_image).convert('RGB')
            
            pil_image.save(image_path)
            self.logger.info(f"Saved chart image: {image_path}")
            
            # Save chart data in CSV format if available
            chart_data = chart_result.get("chart_data", {})
            if chart_data and chart_type == "bar_chart":
                bars = chart_data.get("bars", [])
                if bars:
                    csv_path = chart_dir / "extracted" / f"{base_name}_region{region_id}_bar_data.csv"
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Bar_Index", "X", "Y", "Width", "Height", "Value"])
                        for i, bar in enumerate(bars):
                            bbox = bar["bbox"]
                            writer.writerow([i, bbox[0], bbox[1], bbox[2], bbox[3], bar.get("value", "")])
                    self.logger.info(f"Saved bar chart data: {csv_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save chart result: {str(e)}")
    
    def _save_image_result(self, image_result: Dict[str, Any], base_name: str, region_id: int, region_image: np.ndarray):
        """Save image extraction result and cropped image"""
        try:
            image_dir = self.output_dir / "images"
            
            # Create type-based subdirectories
            image_type = image_result.get("image_type", {}).get("type", "unknown")
            type_dir = image_dir / image_type
            type_dir.mkdir(exist_ok=True)
            
            metadata_dir = image_dir / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            
            # Save JSON metadata
            json_path = metadata_dir / f"{base_name}_region{region_id}_{image_type}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json_safe_result = self._make_json_safe(image_result)
                json.dump(json_safe_result, f, indent=2, ensure_ascii=False)
            
            # Save the cropped image in type-specific directory
            image_path = type_dir / f"{base_name}_region{region_id}_{image_type}.png"
            
            if len(region_image.shape) == 3:
                # Convert BGR to RGB if using OpenCV
                if OPENCV_AVAILABLE:
                    region_image = cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(region_image)
            else:
                pil_image = Image.fromarray(region_image).convert('RGB')
            
            pil_image.save(image_path)
            self.logger.info(f"Saved {image_type} image: {image_path}")
            
            # Save OCR text if available
            ocr_result = image_result.get("ocr_result", {})
            if ocr_result and ocr_result.get("has_text", False):
                text_path = type_dir / f"{base_name}_region{region_id}_{image_type}_text.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(ocr_result.get("text", ""))
                self.logger.info(f"Saved extracted text: {text_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save image result: {str(e)}")
    
    def _create_spatial_relationships(self, layout_result: Dict[str, Any], ocr_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create spatial relationships between elements"""
        try:
            regions = layout_result.get("regions", [])
            
            spatial_map = {
                "region_relationships": [],
                "reading_flow": layout_result.get("reading_order", []),
                "proximity_analysis": {},
                "text_image_correlations": []
            }
            
            # Analyze relationships between regions
            for i, region1 in enumerate(regions):
                for j, region2 in enumerate(regions):
                    if i >= j:  # Avoid duplicate pairs and self-comparison
                        continue
                    
                    relationship = self._analyze_region_relationship(region1, region2)
                    if relationship:
                        spatial_map["region_relationships"].append({
                            "region1_id": region1["id"],
                            "region2_id": region2["id"],
                            "relationship": relationship
                        })
            
            # Correlate with OCR results if available
            if ocr_results:
                text_correlations = self._correlate_with_text(regions, ocr_results)
                spatial_map["text_image_correlations"] = text_correlations
            
            return spatial_map
            
        except Exception as e:
            self.logger.warning(f"Spatial relationship creation failed: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_region_relationship(self, region1: Dict[str, Any], region2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze spatial relationship between two regions"""
        try:
            bbox1 = region1["bbox"]
            bbox2 = region2["bbox"]
            
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            
            # Calculate distances
            center1_x, center1_y = x1 + w1//2, y1 + h1//2
            center2_x, center2_y = x2 + w2//2, y2 + h2//2
            
            distance = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
            
            # Determine relative position
            if abs(center1_y - center2_y) < 20:  # Same row
                if center1_x < center2_x:
                    position = "left_of"
                else:
                    position = "right_of"
            elif abs(center1_x - center2_x) < 20:  # Same column
                if center1_y < center2_y:
                    position = "above"
                else:
                    position = "below"
            else:
                # Diagonal relationship
                if center1_x < center2_x and center1_y < center2_y:
                    position = "top_left_of"
                elif center1_x > center2_x and center1_y < center2_y:
                    position = "top_right_of"
                elif center1_x < center2_x and center1_y > center2_y:
                    position = "bottom_left_of"
                else:
                    position = "bottom_right_of"
            
            # Only return if regions are reasonably close
            if distance < 500:  # Arbitrary threshold
                return {
                    "position": position,
                    "distance": distance,
                    "types": [region1["type"], region2["type"]]
                }
            
            return None
            
        except Exception:
            return None
    
    def _correlate_with_text(self, regions: List[Dict[str, Any]], ocr_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Correlate visual elements with OCR text"""
        correlations = []
        
        try:
            # This is a simplified approach - in practice, you'd need more sophisticated text-image correlation
            full_text = ocr_results.get("full_text", "")
            
            # Look for keywords that might relate to visual elements
            keywords = {
                "table": ["table", "data", "comparison", "vs", "usage", "kwh"],
                "chart": ["chart", "graph", "usage", "monthly", "annual", "percent", "%"],
                "image": ["tip", "save", "energy", "efficient", "home"]
            }
            
            for region in regions:
                region_type = region["type"]
                if region_type in keywords:
                    # Check if related keywords appear in text
                    text_lower = full_text.lower()
                    matches = [kw for kw in keywords[region_type] if kw in text_lower]
                    
                    if matches:
                        correlations.append({
                            "region_id": region["id"],
                            "region_type": region_type,
                            "matching_keywords": matches,
                            "correlation_strength": len(matches) / len(keywords[region_type])
                        })
            
        except Exception as e:
            self.logger.warning(f"Text correlation failed: {str(e)}")
        
        return correlations
    
    def _make_json_safe(self, obj: Any) -> Any:
        """Make object JSON serializable"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_safe(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_safe(item) for item in obj]
        else:
            return obj


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup comprehensive logging"""
    log_dir = Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file or log_dir / "visual_element_extraction.log", encoding='utf-8')
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Visual Element Extraction System - Extract tables, charts, and images from PDF screenshots"
    )
    
    parser.add_argument(
        "input_path",
        help="Path to screenshot image file or directory"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output/visual_element_extraction",
        help="Output directory (default: output/visual_element_extraction)"
    )
    
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Extract only tables"
    )
    
    parser.add_argument(
        "--charts-only", 
        action="store_true",
        help="Extract only charts"
    )
    
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Extract only images"
    )
    
    parser.add_argument(
        "--ocr-results",
        help="Path to OCR results JSON for text correlation"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with intermediate output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else args.log_level
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize processor
        processor = IntegratedVisualProcessor(output_dir=args.output_dir)
        
        input_path = Path(args.input_path)
        
        # Load OCR results if provided
        ocr_results = None
        if args.ocr_results:
            with open(args.ocr_results, 'r', encoding='utf-8') as f:
                ocr_results = json.load(f)
        
        if input_path.is_file():
            # Single file processing
            logger.info(f"Processing single image: {input_path}")
            
            if not input_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                logger.error("Input file must be an image (PNG, JPG, JPEG)")
                sys.exit(1)
            
            result = processor.process_pdf_page_visual(input_path, ocr_results)
            
            # Print summary
            summary = result.get("extraction_summary", {})
            print(f"\n✅ Visual element extraction completed!")
            print(f"   Total regions: {summary.get('total_regions', 0)}")
            print(f"   Tables extracted: {summary.get('tables_extracted', 0)}")
            print(f"   Charts extracted: {summary.get('charts_extracted', 0)}")
            print(f"   Images extracted: {summary.get('images_extracted', 0)}")
            print(f"   Success rate: {summary.get('processing_success_rate', 0):.1%}")
            
        elif input_path.is_dir():
            # Directory processing
            logger.info(f"Processing directory: {input_path}")
            
            image_files = []
            for ext in ['.png', '.jpg', '.jpeg']:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
            if not image_files:
                logger.error("No image files found in directory")
                sys.exit(1)
            
            logger.info(f"Found {len(image_files)} image files")
            
            total_tables = 0
            total_charts = 0  
            total_images = 0
            
            for image_file in image_files:
                logger.info(f"Processing: {image_file}")
                try:
                    result = processor.process_pdf_page_visual(image_file, ocr_results)
                    summary = result.get("extraction_summary", {})
                    
                    total_tables += summary.get('tables_extracted', 0)
                    total_charts += summary.get('charts_extracted', 0)
                    total_images += summary.get('images_extracted', 0)
                    
                    logger.info(f"✅ Completed: {image_file}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed: {image_file} - {str(e)}")
            
            print(f"\n✅ Batch processing completed!")
            print(f"   Files processed: {len(image_files)}")
            print(f"   Total tables extracted: {total_tables}")
            print(f"   Total charts extracted: {total_charts}")
            print(f"   Total images extracted: {total_images}")
            
        else:
            logger.error(f"Invalid input path: {input_path}")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Processing cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()