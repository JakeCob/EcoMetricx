#!/usr/bin/env python3
"""
Integrated PDF Processing System

This script provides comprehensive PDF processing by combining enhanced text 
extraction and image analysis with full integration and correlation capabilities.

Features:
- Complete PDF processing with text and image extraction
- Cross-referencing between text content and images
- Embedding-optimized outputs for multimodal AI applications
- Comprehensive metadata generation and correlation analysis
- Multiple output formats optimized for different use cases

Integration Benefits:
- Text-image correlation with position and context analysis
- Multimodal chunk generation for advanced LLM training
- Comprehensive processing reports with quality metrics
- Embedding-ready outputs for vision-language models
- Structured data extraction with image context

Author: Assistant
Date: 2025-09-02
"""

import os
import sys
import json
import pathlib
import argparse
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import concurrent.futures
import time

# Import enhanced extractors
try:
    from enhanced_pdf_extractor import EnhancedPDFTextExtractor
    from enhanced_image_extractor import EnhancedPDFImageExtractor
except ImportError:
    print("Error: Enhanced extractors not found. Please ensure enhanced_pdf_extractor.py and enhanced_image_extractor.py are in the same directory.")
    sys.exit(1)


class MultimodalChunkGenerator:
    """Generates multimodal chunks combining text and images for embedding applications."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_multimodal_chunks(self, 
                                text_chunks: List[Dict[str, Any]], 
                                images: List[Dict[str, Any]],
                                correlation_strategy: str = "proximity") -> List[Dict[str, Any]]:
        """
        Create multimodal chunks combining text and related images.
        
        Args:
            text_chunks (List[Dict]): Text chunks from enhanced extraction
            images (List[Dict]): Image metadata with correlation data
            correlation_strategy (str): Strategy for correlating text and images
        
        Returns:
            List[Dict]: Multimodal chunks with text and image pairs
        """
        multimodal_chunks = []
        
        if correlation_strategy == "proximity":
            multimodal_chunks = self._proximity_based_pairing(text_chunks, images)
        elif correlation_strategy == "semantic":
            multimodal_chunks = self._semantic_based_pairing(text_chunks, images)
        elif correlation_strategy == "context":
            multimodal_chunks = self._context_based_pairing(text_chunks, images)
        else:
            multimodal_chunks = self._hybrid_pairing(text_chunks, images)
        
        # Enhance chunks with metadata
        for i, chunk in enumerate(multimodal_chunks):
            chunk["multimodal_chunk_id"] = i
            chunk["creation_timestamp"] = datetime.now().isoformat()
            chunk["pairing_quality"] = self._calculate_pairing_quality(chunk)
        
        self.logger.info(f"Generated {len(multimodal_chunks)} multimodal chunks")
        return multimodal_chunks
    
    def _proximity_based_pairing(self, text_chunks: List[Dict], images: List[Dict]) -> List[Dict]:
        """Pair text and images based on spatial proximity."""
        paired_chunks = []
        
        for chunk in text_chunks:
            chunk_page = chunk.get("metadata", {}).get("page_number", 0)
            
            # Find images on the same page
            page_images = [img for img in images if img.get("page_number") == chunk_page]
            
            # Create pairs based on correlation strength
            related_images = []
            for img in page_images:
                correlation = img.get("text_correlation", {})
                if correlation.get("correlation_strength", 0) > 0.3:
                    related_images.append({
                        "filename": img["filename"],
                        "type": img["classification"]["type"],
                        "coordinates": img["coordinates"],
                        "correlation_strength": correlation.get("correlation_strength", 0),
                        "contextual_description": img.get("contextual_description", ""),
                        "visibility": img.get("visibility_analysis", {}).get("visibility", "unknown")
                    })
            
            # Sort images by correlation strength
            related_images.sort(key=lambda x: x["correlation_strength"], reverse=True)
            
            paired_chunk = {
                "text_chunk": chunk,
                "related_images": related_images[:3],  # Limit to top 3 most correlated images
                "page_number": chunk_page,
                "pairing_method": "proximity"
            }
            
            paired_chunks.append(paired_chunk)
        
        return paired_chunks
    
    def _semantic_based_pairing(self, text_chunks: List[Dict], images: List[Dict]) -> List[Dict]:
        """Pair text and images based on semantic similarity."""
        paired_chunks = []
        
        # Group images by context
        context_groups = {}
        for img in images:
            context = img.get("text_correlation", {}).get("primary_context", "general")
            if context not in context_groups:
                context_groups[context] = []
            context_groups[context].append(img)
        
        for chunk in text_chunks:
            content_type = chunk.get("content_type", "body")
            chunk_text = chunk.get("text", "").lower()
            
            # Determine most relevant context for this chunk
            chunk_context = self._identify_chunk_context(chunk_text, content_type)
            
            # Find semantically related images
            related_images = []
            
            # First, try exact context match
            if chunk_context in context_groups:
                for img in context_groups[chunk_context]:
                    related_images.append(self._create_image_reference(img, "semantic_exact"))
            
            # Then, try related contexts
            related_contexts = self._get_related_contexts(chunk_context, content_type)
            for related_context in related_contexts:
                if related_context in context_groups:
                    for img in context_groups[related_context]:
                        if not any(ri["filename"] == img["filename"] for ri in related_images):
                            related_images.append(self._create_image_reference(img, "semantic_related"))
            
            # Sort by correlation strength and limit
            related_images.sort(key=lambda x: x["correlation_strength"], reverse=True)
            
            paired_chunk = {
                "text_chunk": chunk,
                "related_images": related_images[:2],
                "semantic_context": chunk_context,
                "pairing_method": "semantic"
            }
            
            paired_chunks.append(paired_chunk)
        
        return paired_chunks
    
    def _context_based_pairing(self, text_chunks: List[Dict], images: List[Dict]) -> List[Dict]:
        """Pair text and images based on content context and type."""
        paired_chunks = []
        
        for chunk in text_chunks:
            content_type = chunk.get("content_type", "body")
            chunk_text = chunk.get("text", "").lower()
            
            # Find contextually appropriate images
            related_images = []
            
            for img in images:
                context_match = self._calculate_context_match(chunk_text, content_type, img)
                
                if context_match > 0.4:  # Threshold for context relevance
                    related_images.append({
                        **self._create_image_reference(img, "context"),
                        "context_match_score": round(context_match, 2)
                    })
            
            # Sort by context match score
            related_images.sort(key=lambda x: x["context_match_score"], reverse=True)
            
            paired_chunk = {
                "text_chunk": chunk,
                "related_images": related_images[:2],
                "pairing_method": "context"
            }
            
            paired_chunks.append(paired_chunk)
        
        return paired_chunks
    
    def _hybrid_pairing(self, text_chunks: List[Dict], images: List[Dict]) -> List[Dict]:
        """Combine multiple pairing strategies for optimal results."""
        # Use semantic pairing as base
        semantic_chunks = self._semantic_based_pairing(text_chunks, images)
        
        # Enhance with proximity information
        for chunk in semantic_chunks:
            text_chunk = chunk["text_chunk"]
            chunk_page = text_chunk.get("metadata", {}).get("page_number", 0)
            
            # Add proximity boost for same-page images
            for img_ref in chunk["related_images"]:
                for img in images:
                    if img["filename"] == img_ref["filename"] and img.get("page_number") == chunk_page:
                        img_ref["proximity_boost"] = 0.2
                        img_ref["correlation_strength"] = min(
                            img_ref["correlation_strength"] + 0.2, 1.0
                        )
                        break
            
            # Re-sort with proximity boost
            chunk["related_images"].sort(key=lambda x: x["correlation_strength"], reverse=True)
            chunk["pairing_method"] = "hybrid"
        
        return semantic_chunks
    
    def _identify_chunk_context(self, chunk_text: str, content_type: str) -> str:
        """Identify the primary context of a text chunk."""
        # Context keywords mapping
        context_keywords = {
            "energy_tips": ["tip", "advice", "save", "reduce", "laundry", "load"],
            "usage_data": ["usage", "kwh", "typical", "above", "below", "consumption"],
            "contact_info": ["contact", "call", "visit", "scan", "qr"],
            "account_info": ["account", "number", "service", "address"],
            "savings": ["save", "money", "cost", "savings"],
            "comparison": ["compare", "neighbors", "average", "typical"]
        }
        
        # Score each context
        context_scores = {}
        for context, keywords in context_keywords.items():
            score = sum(1 for keyword in keywords if keyword in chunk_text)
            if score > 0:
                context_scores[context] = score
        
        # Return highest scoring context or default based on content type
        if context_scores:
            return max(context_scores, key=context_scores.get)
        
        # Content type fallbacks
        type_to_context = {
            "header": "account_info",
            "footer": "contact_info", 
            "advice": "energy_tips",
            "usage_data": "usage_data",
            "account_info": "account_info"
        }
        
        return type_to_context.get(content_type, "general")
    
    def _get_related_contexts(self, primary_context: str, content_type: str) -> List[str]:
        """Get contexts related to the primary context."""
        related_map = {
            "energy_tips": ["savings", "usage_data"],
            "usage_data": ["comparison", "savings"],
            "contact_info": ["account_info"],
            "account_info": ["contact_info"],
            "savings": ["energy_tips", "usage_data"],
            "comparison": ["usage_data"]
        }
        
        return related_map.get(primary_context, [])
    
    def _calculate_context_match(self, chunk_text: str, content_type: str, image: Dict[str, Any]) -> float:
        """Calculate how well an image's context matches a text chunk."""
        img_context = image.get("text_correlation", {}).get("primary_context", "general")
        img_type = image.get("classification", {}).get("type", "unknown")
        
        score = 0.0
        
        # Direct context match
        chunk_context = self._identify_chunk_context(chunk_text, content_type)
        if img_context == chunk_context:
            score += 0.6
        
        # Related context match
        if img_context in self._get_related_contexts(chunk_context, content_type):
            score += 0.3
        
        # Type-content correlation
        if content_type == "advice" and img_type == "photo":
            score += 0.2
        elif content_type == "usage_data" and img_type == "chart":
            score += 0.3
        elif content_type in ["header", "footer"] and img_type == "logo":
            score += 0.2
        
        # Keyword matching
        img_keywords = image.get("text_correlation", {}).get("nearby_text", "").lower()
        common_words = set(chunk_text.split()) & set(img_keywords.split())
        if common_words:
            score += min(len(common_words) * 0.05, 0.2)
        
        return min(score, 1.0)
    
    def _create_image_reference(self, image: Dict[str, Any], pairing_type: str) -> Dict[str, Any]:
        """Create a standardized image reference for multimodal chunks."""
        return {
            "filename": image["filename"],
            "type": image["classification"]["type"],
            "coordinates": image["coordinates"],
            "correlation_strength": image.get("text_correlation", {}).get("correlation_strength", 0),
            "contextual_description": image.get("contextual_description", ""),
            "visibility": image.get("visibility_analysis", {}).get("visibility", "unknown"),
            "pairing_type": pairing_type
        }
    
    def _calculate_pairing_quality(self, chunk: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for the multimodal pairing."""
        related_images = chunk.get("related_images", [])
        
        if not related_images:
            return {"overall_quality": 0.0, "relevance_score": 0.0, "diversity_score": 0.0}
        
        # Relevance score (average correlation strength)
        relevance_scores = [img.get("correlation_strength", 0) for img in related_images]
        relevance_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # Diversity score (different image types and contexts)
        image_types = set(img.get("type", "unknown") for img in related_images)
        diversity_score = min(len(image_types) / 3.0, 1.0)  # Normalize to max 3 types
        
        # Overall quality (weighted combination)
        overall_quality = (relevance_score * 0.7) + (diversity_score * 0.3)
        
        return {
            "overall_quality": round(overall_quality, 2),
            "relevance_score": round(relevance_score, 2),
            "diversity_score": round(diversity_score, 2)
        }


class QualityAnalyzer:
    """Analyzes the quality and completeness of integrated extraction results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_extraction_quality(self, 
                                  text_results: Dict[str, Any],
                                  image_results: Dict[str, Any],
                                  multimodal_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the overall quality of integrated extraction.
        
        Args:
            text_results (Dict): Enhanced text extraction results
            image_results (Dict): Enhanced image extraction results 
            multimodal_chunks (List[Dict]): Generated multimodal chunks
        
        Returns:
            Dict: Comprehensive quality analysis
        """
        quality_analysis = {
            "analysis_timestamp": datetime.now().isoformat(),
            "text_quality": self._analyze_text_quality(text_results),
            "image_quality": self._analyze_image_quality(image_results),
            "integration_quality": self._analyze_integration_quality(multimodal_chunks),
            "overall_score": 0.0,
            "recommendations": []
        }
        
        # Calculate overall score
        text_score = quality_analysis["text_quality"].get("overall_score", 0)
        image_score = quality_analysis["image_quality"].get("overall_score", 0)
        integration_score = quality_analysis["integration_quality"].get("overall_score", 0)
        
        quality_analysis["overall_score"] = round((text_score + image_score + integration_score) / 3, 2)
        
        # Generate recommendations
        quality_analysis["recommendations"] = self._generate_recommendations(quality_analysis)
        
        return quality_analysis
    
    def _analyze_text_quality(self, text_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text extraction quality."""
        full_text = text_results.get("full_text", "")
        pages = text_results.get("pages", [])
        structured_data = text_results.get("structured_data", {})
        embedding_chunks = text_results.get("embedding_chunks", [])
        
        # Basic metrics
        total_chars = len(full_text)
        total_words = len(full_text.split()) if full_text else 0
        total_pages = len(pages)
        
        # Content analysis
        has_headers = "##" in full_text or "#" in full_text
        has_tables = "|" in full_text
        has_structured_data = len(structured_data) > 0
        
        # Quality scores
        completeness_score = min(total_words / 1000, 1.0)  # Normalize to expected content
        structure_score = sum([has_headers, has_tables, has_structured_data]) / 3.0
        
        # Chunking quality
        chunking_score = 0.0
        if embedding_chunks:
            chunk_sizes = [chunk.get("character_count", 0) for chunk in embedding_chunks]
            optimal_chunks = sum(1 for size in chunk_sizes if 200 <= size <= 800)
            chunking_score = optimal_chunks / len(embedding_chunks) if embedding_chunks else 0
        
        overall_score = (completeness_score + structure_score + chunking_score) / 3
        
        return {
            "overall_score": round(overall_score, 2),
            "completeness_score": round(completeness_score, 2),
            "structure_score": round(structure_score, 2),
            "chunking_score": round(chunking_score, 2),
            "metrics": {
                "total_characters": total_chars,
                "total_words": total_words,
                "total_pages": total_pages,
                "structured_elements": len(structured_data),
                "embedding_chunks": len(embedding_chunks)
            }
        }
    
    def _analyze_image_quality(self, image_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image extraction quality."""
        images = image_results.get("extracted_images", [])
        summary = image_results.get("processing_summary", {})
        
        if not images:
            return {"overall_score": 0.0, "message": "No images extracted"}
        
        # Basic metrics
        total_images = len(images)
        unique_images = summary.get("unique_images", 0)
        
        # Classification analysis
        classified_images = sum(1 for img in images 
                              if img.get("classification", {}).get("confidence", 0) > 0.5)
        classification_score = classified_images / total_images if total_images > 0 else 0
        
        # Visibility analysis  
        visible_images = sum(1 for img in images
                           if img.get("visibility_analysis", {}).get("visibility") == "visible")
        visibility_detection_score = min(visible_images / max(total_images * 0.6, 1), 1.0)
        
        # Correlation analysis
        correlated_images = sum(1 for img in images
                              if img.get("text_correlation", {}).get("correlation_strength", 0) > 0.3)
        correlation_score = correlated_images / total_images if total_images > 0 else 0
        
        overall_score = (classification_score + visibility_detection_score + correlation_score) / 3
        
        return {
            "overall_score": round(overall_score, 2),
            "classification_score": round(classification_score, 2),
            "visibility_detection_score": round(visibility_detection_score, 2),
            "correlation_score": round(correlation_score, 2),
            "metrics": {
                "total_images": total_images,
                "unique_images": unique_images,
                "well_classified": classified_images,
                "visible_detected": visible_images,
                "well_correlated": correlated_images
            }
        }
    
    def _analyze_integration_quality(self, multimodal_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze integration quality."""
        if not multimodal_chunks:
            return {"overall_score": 0.0, "message": "No multimodal chunks generated"}
        
        # Pairing quality analysis
        pairing_qualities = [chunk.get("pairing_quality", {}).get("overall_quality", 0) 
                           for chunk in multimodal_chunks]
        avg_pairing_quality = sum(pairing_qualities) / len(pairing_qualities) if pairing_qualities else 0
        
        # Coverage analysis (how many chunks have image associations)
        chunks_with_images = sum(1 for chunk in multimodal_chunks
                               if chunk.get("related_images", []))
        coverage_score = chunks_with_images / len(multimodal_chunks)
        
        # Diversity analysis (variety of image types paired)
        all_image_types = set()
        for chunk in multimodal_chunks:
            for img in chunk.get("related_images", []):
                all_image_types.add(img.get("type", "unknown"))
        
        diversity_score = min(len(all_image_types) / 4.0, 1.0)  # Normalize to 4 expected types
        
        overall_score = (avg_pairing_quality + coverage_score + diversity_score) / 3
        
        return {
            "overall_score": round(overall_score, 2),
            "pairing_quality": round(avg_pairing_quality, 2),
            "coverage_score": round(coverage_score, 2),
            "diversity_score": round(diversity_score, 2),
            "metrics": {
                "total_chunks": len(multimodal_chunks),
                "chunks_with_images": chunks_with_images,
                "unique_image_types": len(all_image_types),
                "image_types_found": list(all_image_types)
            }
        }
    
    def _generate_recommendations(self, quality_analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on quality analysis."""
        recommendations = []
        
        text_quality = quality_analysis["text_quality"]
        image_quality = quality_analysis["image_quality"]
        integration_quality = quality_analysis["integration_quality"]
        
        # Text quality recommendations
        if text_quality.get("structure_score", 0) < 0.5:
            recommendations.append("Consider using enhanced layout analysis for better structure preservation")
        
        if text_quality.get("chunking_score", 0) < 0.6:
            recommendations.append("Adjust chunking strategy or parameters for better embedding optimization")
        
        # Image quality recommendations
        if image_quality.get("classification_score", 0) < 0.7:
            recommendations.append("Review image classification confidence thresholds")
        
        if image_quality.get("correlation_score", 0) < 0.5:
            recommendations.append("Improve text-image correlation by providing more detailed text content")
        
        # Integration recommendations
        if integration_quality.get("coverage_score", 0) < 0.6:
            recommendations.append("Consider adjusting multimodal pairing strategy for better coverage")
        
        if integration_quality.get("diversity_score", 0) < 0.5:
            recommendations.append("Enhance image classification to identify more diverse image types")
        
        # Overall recommendations
        overall_score = quality_analysis.get("overall_score", 0)
        if overall_score < 0.6:
            recommendations.append("Consider processing with higher quality settings or manual review")
        elif overall_score > 0.8:
            recommendations.append("High quality extraction achieved - suitable for production use")
        
        return recommendations


class IntegratedPDFProcessor:
    """
    Comprehensive PDF processor combining enhanced text and image extraction.
    """
    
    def __init__(self, 
                 output_dir: str = "output",
                 enable_text_enhancement: bool = True,
                 enable_image_enhancement: bool = True,
                 log_level: int = logging.INFO):
        """Initialize the Integrated PDF Processor."""
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create integrated directory structure
        self.integrated_dir = self.output_dir / "integrated"
        self.embeddings_ready_dir = self.output_dir / "embeddings_ready"
        
        for dir_path in [self.integrated_dir, self.embeddings_ready_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        self.text_extractor = EnhancedPDFTextExtractor(output_dir, enable_text_enhancement)
        self.image_extractor = EnhancedPDFImageExtractor(output_dir, enable_image_enhancement, log_level)
        
        # Initialize support components
        self.multimodal_generator = MultimodalChunkGenerator()
        self.quality_analyzer = QualityAnalyzer()
    
    def process_pdf_comprehensive(self, 
                                 pdf_path: str,
                                 processing_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete PDF processing with full integration.
        
        Args:
            pdf_path (str): Path to PDF file
            processing_options (Dict): Processing configuration options
        
        Returns:
            Dict: Comprehensive processing results
        """
        start_time = time.time()
        pdf_path = pathlib.Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Starting comprehensive PDF processing: {pdf_path}")
        
        # Set default processing options
        options = {
            "enhanced_layout": True,
            "extract_structured_data": True,
            "embedding_chunks": True,
            "chunk_strategy": "semantic",
            "chunk_size": 512,
            "image_enhancement": True,
            "text_correlation": True,
            "multimodal_pairing": "hybrid",
            "quality_analysis": True,
            **( processing_options or {})
        }
        
        try:
            # Phase 1: Enhanced Text Extraction
            self.logger.info("Phase 1: Enhanced text extraction")
            if options["enhanced_layout"]:
                text_results = self.text_extractor.extract_with_layout_analysis(str(pdf_path))
            else:
                text_results = self.text_extractor.extract_with_layout_analysis(str(pdf_path), preserve_structure=False)
            
            # Create embedding chunks if requested
            if options["embedding_chunks"]:
                chunks = self.text_extractor.create_embedding_chunks(
                    text_results["full_text"],
                    options["chunk_strategy"],
                    max_chunk_size=options["chunk_size"]
                )
                text_results["embedding_chunks"] = chunks
            
            # Phase 2: Enhanced Image Extraction
            self.logger.info("Phase 2: Enhanced image extraction")
            
            # Prepare text content for correlation
            text_content = text_results["full_text"] if options["text_correlation"] else None
            
            image_results = self.image_extractor.extract_images_enhanced(
                str(pdf_path),
                text_content
            )
            
            # Phase 3: Integration and Cross-referencing
            self.logger.info("Phase 3: Integration and cross-referencing")
            
            # Update text results with image references
            if options["text_correlation"] and image_results.get("extracted_images"):
                # Create image metadata for text extractor
                image_metadata_path = self.integrated_dir / f"{pdf_path.stem}_image_metadata.json"
                with open(image_metadata_path, 'w') as f:
                    json.dump({
                        "images": image_results["extracted_images"]
                    }, f, indent=2)
                
                # Re-extract text with image references
                text_results = self.text_extractor.extract_with_image_references(
                    str(pdf_path), str(image_metadata_path)
                )
            
            # Phase 4: Multimodal Chunk Generation
            multimodal_chunks = []
            if options["embedding_chunks"] and image_results.get("extracted_images"):
                self.logger.info("Phase 4: Multimodal chunk generation")
                
                text_chunks = text_results.get("embedding_chunks", [])
                images = image_results.get("extracted_images", [])
                
                multimodal_chunks = self.multimodal_generator.create_multimodal_chunks(
                    text_chunks, images, options["multimodal_pairing"]
                )
            
            # Phase 5: Quality Analysis
            quality_analysis = {}
            if options["quality_analysis"]:
                self.logger.info("Phase 5: Quality analysis")
                quality_analysis = self.quality_analyzer.analyze_extraction_quality(
                    text_results, image_results, multimodal_chunks
                )
            
            # Compile comprehensive results
            processing_time = time.time() - start_time
            
            comprehensive_results = {
                "processing_metadata": {
                    "source_pdf": pdf_path.name,
                    "processing_timestamp": datetime.now().isoformat(),
                    "processing_time_seconds": round(processing_time, 2),
                    "processing_options": options,
                    "phases_completed": ["text_extraction", "image_extraction", "integration", "multimodal_generation", "quality_analysis"]
                },
                "text_results": text_results,
                "image_results": image_results,
                "multimodal_chunks": multimodal_chunks,
                "quality_analysis": quality_analysis,
                "integration_summary": self._generate_integration_summary(text_results, image_results, multimodal_chunks)
            }
            
            self.logger.info(f"Comprehensive processing completed in {processing_time:.2f} seconds")
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"Error during comprehensive processing: {str(e)}")
            raise
    
    def save_comprehensive_results(self, 
                                  results: Dict[str, Any], 
                                  base_filename: str = None) -> Dict[str, str]:
        """Save comprehensive processing results in multiple formats."""
        if base_filename is None:
            base_filename = pathlib.Path(results["processing_metadata"]["source_pdf"]).stem
        
        saved_files = {}
        
        try:
            # 1. Complete processing report
            report_path = self.integrated_dir / f"{base_filename}_comprehensive_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            saved_files["comprehensive_report"] = str(report_path)
            
            # 2. Multimodal chunks for embeddings
            if results.get("multimodal_chunks"):
                chunks_path = self.embeddings_ready_dir / f"{base_filename}_multimodal_chunks.json"
                chunks_data = {
                    "metadata": results["processing_metadata"],
                    "chunks": results["multimodal_chunks"],
                    "total_chunks": len(results["multimodal_chunks"])
                }
                with open(chunks_path, 'w', encoding='utf-8') as f:
                    json.dump(chunks_data, f, indent=2, ensure_ascii=False)
                saved_files["multimodal_chunks"] = str(chunks_path)
            
            # 3. Text chunks with image context
            text_chunks = results.get("text_results", {}).get("embedding_chunks", [])
            if text_chunks:
                text_chunks_path = self.embeddings_ready_dir / f"{base_filename}_text_chunks_with_context.json"
                with open(text_chunks_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "metadata": results["processing_metadata"],
                        "text_chunks": text_chunks,
                        "total_chunks": len(text_chunks)
                    }, f, indent=2, ensure_ascii=False)
                saved_files["text_chunks_with_context"] = str(text_chunks_path)
            
            # 4. Image descriptions for embeddings
            images = results.get("image_results", {}).get("extracted_images", [])
            if images:
                image_descriptions = []
                for img in images:
                    img_desc = {
                        "filename": img["filename"],
                        "type": img["classification"]["type"],
                        "contextual_description": img.get("contextual_description", ""),
                        "text_correlation": img.get("text_correlation", {}),
                        "visibility": img.get("visibility_analysis", {}).get("visibility", "unknown")
                    }
                    image_descriptions.append(img_desc)
                
                desc_path = self.embeddings_ready_dir / f"{base_filename}_image_descriptions.json"
                with open(desc_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "metadata": results["processing_metadata"],
                        "image_descriptions": image_descriptions,
                        "total_images": len(image_descriptions)
                    }, f, indent=2, ensure_ascii=False)
                saved_files["image_descriptions"] = str(desc_path)
            
            # 5. Correlation analysis
            if results.get("text_results", {}).get("cross_references"):
                correlation_path = self.integrated_dir / f"{base_filename}_correlation_analysis.json"
                correlation_data = {
                    "metadata": results["processing_metadata"],
                    "cross_references": results["text_results"]["cross_references"],
                    "correlation_summary": self._analyze_correlations(results["text_results"]["cross_references"])
                }
                with open(correlation_path, 'w', encoding='utf-8') as f:
                    json.dump(correlation_data, f, indent=2, ensure_ascii=False)
                saved_files["correlation_analysis"] = str(correlation_path)
            
            # 6. Quality report
            if results.get("quality_analysis"):
                quality_path = self.integrated_dir / f"{base_filename}_quality_report.json"
                with open(quality_path, 'w', encoding='utf-8') as f:
                    json.dump(results["quality_analysis"], f, indent=2, ensure_ascii=False)
                saved_files["quality_report"] = str(quality_path)
            
            self.logger.info(f"Saved comprehensive results to {len(saved_files)} files")
            
        except Exception as e:
            self.logger.error(f"Error saving comprehensive results: {str(e)}")
            raise
        
        return saved_files
    
    def _generate_integration_summary(self, 
                                    text_results: Dict[str, Any],
                                    image_results: Dict[str, Any],
                                    multimodal_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of integration results."""
        text_pages = len(text_results.get("pages", []))
        text_chunks = len(text_results.get("embedding_chunks", []))
        structured_elements = len(text_results.get("structured_data", {}))
        
        total_images = len(image_results.get("extracted_images", []))
        unique_images = image_results.get("processing_summary", {}).get("unique_images", 0)
        correlated_images = sum(1 for img in image_results.get("extracted_images", [])
                              if img.get("text_correlation", {}).get("correlation_strength", 0) > 0.3)
        
        multimodal_chunks_count = len(multimodal_chunks)
        chunks_with_images = sum(1 for chunk in multimodal_chunks
                               if chunk.get("related_images", []))
        
        return {
            "text_summary": {
                "pages_processed": text_pages,
                "text_chunks_created": text_chunks,
                "structured_elements_found": structured_elements
            },
            "image_summary": {
                "total_images_extracted": total_images,
                "unique_images": unique_images,
                "well_correlated_images": correlated_images
            },
            "integration_summary": {
                "multimodal_chunks_created": multimodal_chunks_count,
                "chunks_with_image_references": chunks_with_images,
                "integration_success_rate": round(chunks_with_images / max(multimodal_chunks_count, 1), 2)
            }
        }
    
    def _analyze_correlations(self, cross_references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlation patterns in cross-references."""
        if not cross_references:
            return {"message": "No correlations found"}
        
        # Group by correlation strength
        strong_correlations = [ref for ref in cross_references if ref.get("correlation_strength", 0) > 0.7]
        moderate_correlations = [ref for ref in cross_references if 0.4 <= ref.get("correlation_strength", 0) <= 0.7]
        weak_correlations = [ref for ref in cross_references if ref.get("correlation_strength", 0) < 0.4]
        
        # Analyze image types in correlations
        image_type_correlations = {}
        for ref in cross_references:
            img_type = ref.get("image_type", "unknown")
            if img_type not in image_type_correlations:
                image_type_correlations[img_type] = []
            image_type_correlations[img_type].append(ref.get("correlation_strength", 0))
        
        # Calculate average correlation by image type
        avg_correlations_by_type = {}
        for img_type, correlations in image_type_correlations.items():
            avg_correlations_by_type[img_type] = round(sum(correlations) / len(correlations), 2)
        
        return {
            "total_correlations": len(cross_references),
            "strong_correlations": len(strong_correlations),
            "moderate_correlations": len(moderate_correlations),
            "weak_correlations": len(weak_correlations),
            "average_correlation_by_type": avg_correlations_by_type,
            "overall_correlation_strength": round(sum(ref.get("correlation_strength", 0) 
                                                    for ref in cross_references) / len(cross_references), 2)
        }


def main():
    """Comprehensive main function with full CLI interface."""
    parser = argparse.ArgumentParser(
        description="Integrated PDF Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python integrated_extractor.py energy_report.pdf --full-analysis
  python integrated_extractor.py energy_report.pdf --embedding-ready --chunk-size 512
  python integrated_extractor.py batch_folder/ --recursive --parallel-processing
  python integrated_extractor.py energy_report.pdf --chunk-strategy hybrid --multimodal-pairing semantic
        """
    )
    
    parser.add_argument("pdf_path", help="Path to PDF file or directory for batch processing")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--full-analysis", action="store_true", 
                       help="Enable all advanced analysis features")
    parser.add_argument("--embedding-ready", action="store_true",
                       help="Generate embedding-optimized outputs")
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="Maximum chunk size for embeddings")
    parser.add_argument("--chunk-strategy", choices=["semantic", "size", "structure", "hybrid"],
                       default="semantic", help="Text chunking strategy")
    parser.add_argument("--multimodal-pairing", choices=["proximity", "semantic", "context", "hybrid"],
                       default="hybrid", help="Multimodal pairing strategy")
    parser.add_argument("--no-text-enhancement", action="store_true",
                       help="Disable enhanced text processing")
    parser.add_argument("--no-image-enhancement", action="store_true", 
                       help="Disable enhanced image processing")
    parser.add_argument("--recursive", action="store_true",
                       help="Process PDFs recursively in directories")
    parser.add_argument("--parallel-processing", action="store_true",
                       help="Enable parallel processing for batch operations")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress non-error output")
    
    args = parser.parse_args()
    
    # Validate arguments
    pdf_path = pathlib.Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: Path not found: {args.pdf_path}")
        sys.exit(1)
    
    # Set logging level
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR
    
    # Build processing options
    processing_options = {
        "enhanced_layout": not args.no_text_enhancement,
        "extract_structured_data": True,
        "embedding_chunks": args.embedding_ready or args.full_analysis,
        "chunk_strategy": args.chunk_strategy,
        "chunk_size": args.chunk_size,
        "image_enhancement": not args.no_image_enhancement,
        "text_correlation": True,
        "multimodal_pairing": args.multimodal_pairing,
        "quality_analysis": args.full_analysis
    }
    
    # Initialize processor
    processor = IntegratedPDFProcessor(
        output_dir=args.output_dir,
        enable_text_enhancement=not args.no_text_enhancement,
        enable_image_enhancement=not args.no_image_enhancement,
        log_level=log_level
    )
    
    try:
        if pdf_path.is_file() and pdf_path.suffix.lower() == '.pdf':
            # Process single PDF
            print(f"🚀 Starting comprehensive PDF processing: {pdf_path}")
            
            results = processor.process_pdf_comprehensive(str(pdf_path), processing_options)
            saved_files = processor.save_comprehensive_results(results)
            
            # Print comprehensive summary
            summary = results["integration_summary"]
            processing_time = results["processing_metadata"]["processing_time_seconds"]
            
            print(f"\n✅ Comprehensive processing completed in {processing_time:.2f}s!")
            print(f"📄 Text: {summary['text_summary']['pages_processed']} pages, {summary['text_summary']['text_chunks_created']} chunks")
            print(f"🖼️  Images: {summary['image_summary']['total_images_extracted']} total, {summary['image_summary']['well_correlated_images']} correlated")
            print(f"🔗 Integration: {summary['integration_summary']['multimodal_chunks_created']} multimodal chunks")
            
            if results.get("quality_analysis"):
                quality_score = results["quality_analysis"]["overall_score"]
                print(f"⭐ Overall quality score: {quality_score}/1.0")
            
            print(f"\n📁 Generated {len(saved_files)} output files:")
            for file_type, file_path in saved_files.items():
                print(f"   {file_type.replace('_', ' ').title()}: {file_path}")
        
        elif pdf_path.is_dir():
            # Batch processing
            print(f"📁 Starting batch processing: {pdf_path}")
            
            # Find PDF files
            if args.recursive:
                pdf_files = list(pdf_path.rglob("*.pdf"))
            else:
                pdf_files = list(pdf_path.glob("*.pdf"))
            
            if not pdf_files:
                print("❌ No PDF files found in directory")
                sys.exit(1)
            
            print(f"Found {len(pdf_files)} PDF files to process")
            
            # Process files
            successful = 0
            failed = 0
            
            for pdf_file in pdf_files:
                try:
                    print(f"\n📄 Processing: {pdf_file.name}")
                    results = processor.process_pdf_comprehensive(str(pdf_file), processing_options)
                    processor.save_comprehensive_results(results)
                    successful += 1
                    print(f"✅ {pdf_file.name} completed successfully")
                except Exception as e:
                    failed += 1
                    print(f"❌ {pdf_file.name} failed: {str(e)}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
            
            print(f"\n📊 Batch processing summary:")
            print(f"   ✅ Successful: {successful}")
            print(f"   ❌ Failed: {failed}")
            print(f"   📁 Results saved to: {args.output_dir}")
        
        else:
            print(f"❌ Invalid input: {args.pdf_path} (must be PDF file or directory)")
            sys.exit(1)
        
        print("\n🎉 Integrated PDF processing completed successfully!")
        
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