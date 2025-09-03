"""
Enhanced Output Manager for EcoMetricx
Provides unified, query-optimized output structure for all extraction methods.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import pandas as pd


class EcoMetricxOutputManager:
    """
    Comprehensive output manager for organizing all EcoMetricx extraction results.
    Optimized for query systems and demonstration purposes.
    """
    
    def __init__(self, base_output_dir: str = "output", session_id: Optional[str] = None):
        """
        Initialize the output manager with enhanced structure.
        
        Args:
            base_output_dir: Base directory for all outputs
            session_id: Optional session ID, auto-generated if None
        """
        self.base_dir = Path(base_output_dir)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_timestamp = datetime.now().isoformat()
        
        # Initialize directory structure
        self._setup_directory_structure()
        
        # Initialize session metadata
        self.session_metadata = {
            'session_id': self.session_id,
            'created_at': self.session_timestamp,
            'documents_processed': [],
            'extraction_methods_used': [],
            'total_files_created': 0,
            'processing_statistics': {}
        }
        
        # Save initial session metadata
        self._save_session_metadata()
    
    def _setup_directory_structure(self):
        """Create the complete directory structure."""
        
        # Main structure directories
        directories = [
            # Session level
            self.base_dir / "session_metadata",
            
            # Documents will be created per-document
            self.base_dir / "documents",
            
            # Query-optimized formats
            self.base_dir / "queryable" / "search_indices",
            self.base_dir / "queryable" / "database_ready",
            self.base_dir / "queryable" / "api_ready",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _get_timestamp(self) -> str:
        """Get standardized timestamp string."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize filename for cross-platform compatibility."""
        import re
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Remove extra underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized.strip('_')
    
    def _save_session_metadata(self):
        """Save session metadata to file."""
        session_file = self.base_dir / "session_metadata" / f"{self.session_id}_session_summary.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(self.session_metadata, f, indent=2, ensure_ascii=False)
    
    def register_document(self, document_path: str) -> str:
        """
        Register a new document for processing and create its directory structure.
        
        Args:
            document_path: Path to the source document
            
        Returns:
            Document ID for use in subsequent operations
        """
        doc_path = Path(document_path)
        doc_name = self._sanitize_filename(doc_path.stem)
        
        # Create document-specific directories
        doc_base = self.base_dir / "documents" / doc_name
        
        directories = [
            # Enhanced PDF extraction
            doc_base / "enhanced_pdf" / "text",
            doc_base / "enhanced_pdf" / "metadata",
            
            # Visual OCR extraction  
            doc_base / "visual_ocr" / "screenshots",
            doc_base / "visual_ocr" / "text",
            doc_base / "visual_ocr" / "metadata",
            
            # Visual elements extraction
            doc_base / "visual_elements" / "extracted" / "tables",
            doc_base / "visual_elements" / "extracted" / "charts", 
            doc_base / "visual_elements" / "extracted" / "images",
            doc_base / "visual_elements" / "metadata",
            
            # Enhanced images extraction
            doc_base / "enhanced_images" / "organized" / "by_type" / "chart",
            doc_base / "enhanced_images" / "organized" / "by_type" / "logo",
            doc_base / "enhanced_images" / "organized" / "by_type" / "qr_code",
            doc_base / "enhanced_images" / "organized" / "by_type" / "photo",
            doc_base / "enhanced_images" / "organized" / "by_visibility" / "visible",
            doc_base / "enhanced_images" / "organized" / "by_visibility" / "embedded", 
            doc_base / "enhanced_images" / "organized" / "by_visibility" / "background",
            doc_base / "enhanced_images" / "enhanced",
            doc_base / "enhanced_images" / "metadata",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create document metadata
        doc_metadata = {
            'document_id': doc_name,
            'original_path': str(doc_path.absolute()),
            'registered_at': datetime.now().isoformat(),
            'file_size': doc_path.stat().st_size if doc_path.exists() else 0,
            'processing_status': 'registered',
            'extractions_completed': []
        }
        
        # Save document metadata
        doc_metadata_file = doc_base / "document_metadata.json"
        with open(doc_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(doc_metadata, f, indent=2, ensure_ascii=False)
        
        # Update session metadata
        if doc_name not in self.session_metadata['documents_processed']:
            self.session_metadata['documents_processed'].append(doc_name)
        self._save_session_metadata()
        
        return doc_name
    
    def save_enhanced_pdf_extraction(self, document_id: str, extraction_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Save enhanced PDF extraction results with proper organization.
        
        Args:
            document_id: Document identifier
            extraction_result: Results from enhanced PDF extraction
            
        Returns:
            Dictionary of saved file paths
        """
        timestamp = self._get_timestamp()
        doc_base = self.base_dir / "documents" / document_id / "enhanced_pdf"
        
        saved_files = {}
        
        # Save full text
        if 'full_text' in extraction_result:
            text_file = doc_base / "text" / f"{document_id}_{timestamp}_full_text.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(extraction_result['full_text'])
            saved_files['full_text'] = str(text_file)
        
        # Save structured data
        if 'structured_data' in extraction_result:
            structured_file = doc_base / "text" / f"{document_id}_{timestamp}_structured_data.json"
            with open(structured_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_result['structured_data'], f, indent=2, ensure_ascii=False)
            saved_files['structured_data'] = str(structured_file)
        
        # Save layout analysis
        if 'layout_analysis' in extraction_result:
            layout_file = doc_base / "text" / f"{document_id}_{timestamp}_layout_analysis.json"
            with open(layout_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_result['layout_analysis'], f, indent=2, ensure_ascii=False)
            saved_files['layout_analysis'] = str(layout_file)
        
        # Create extraction report
        extraction_report = {
            'extraction_method': 'enhanced_pdf',
            'document_id': document_id,
            'timestamp': timestamp,
            'processing_time': extraction_result.get('processing_time', 0),
            'text_length': len(extraction_result.get('full_text', '')),
            'confidence_score': extraction_result.get('confidence', 95),
            'pages_processed': extraction_result.get('total_pages', 0),
            'structured_elements_found': len(extraction_result.get('structured_data', {})),
            'saved_files': saved_files
        }
        
        report_file = doc_base / "metadata" / f"{document_id}_{timestamp}_extraction_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_report, f, indent=2, ensure_ascii=False)
        saved_files['extraction_report'] = str(report_file)
        
        # Update method usage tracking
        if 'enhanced_pdf' not in self.session_metadata['extraction_methods_used']:
            self.session_metadata['extraction_methods_used'].append('enhanced_pdf')
        
        self.session_metadata['total_files_created'] += len(saved_files)
        self._save_session_metadata()
        
        return saved_files
    
    def save_visual_ocr_extraction(self, document_id: str, extraction_result: Dict[str, Any], 
                                 screenshots: Optional[List[Path]] = None) -> Dict[str, str]:
        """
        Save visual OCR extraction results with proper organization.
        
        Args:
            document_id: Document identifier
            extraction_result: Results from visual OCR extraction
            screenshots: Optional list of screenshot paths
            
        Returns:
            Dictionary of saved file paths
        """
        timestamp = self._get_timestamp()
        doc_base = self.base_dir / "documents" / document_id / "visual_ocr"
        
        saved_files = {}
        
        # Save screenshots with manifest
        if screenshots:
            screenshot_manifest = []
            for i, screenshot in enumerate(screenshots):
                if screenshot.exists():
                    new_name = f"{document_id}_page_{i}_{timestamp}.png"
                    target_path = doc_base / "screenshots" / new_name
                    shutil.copy2(screenshot, target_path)
                    saved_files[f'screenshot_{i}'] = str(target_path)
                    screenshot_manifest.append({
                        'page_number': i,
                        'filename': new_name,
                        'original_path': str(screenshot),
                        'file_size': target_path.stat().st_size
                    })
            
            # Save screenshot manifest
            manifest_file = doc_base / "screenshots" / "screenshots_manifest.json"
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(screenshot_manifest, f, indent=2, ensure_ascii=False)
            saved_files['screenshot_manifest'] = str(manifest_file)
        
        # Save OCR text
        if 'full_text' in extraction_result:
            text_file = doc_base / "text" / f"{document_id}_{timestamp}_ocr_text.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(extraction_result['full_text'])
            saved_files['ocr_text'] = str(text_file)
        
        # Save confidence scores
        if 'confidence_scores' in extraction_result:
            confidence_file = doc_base / "text" / f"{document_id}_{timestamp}_confidence_scores.json"
            with open(confidence_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_result['confidence_scores'], f, indent=2, ensure_ascii=False)
            saved_files['confidence_scores'] = str(confidence_file)
        
        # Create OCR report
        ocr_report = {
            'extraction_method': 'visual_ocr',
            'document_id': document_id,
            'timestamp': timestamp,
            'processing_time': extraction_result.get('processing_time', 0),
            'text_length': len(extraction_result.get('full_text', '')),
            'average_confidence': extraction_result.get('average_confidence', 0),
            'pages_processed': extraction_result.get('total_pages', 0),
            'dpi': extraction_result.get('dpi', 300),
            'screenshots_saved': len(screenshots) if screenshots else 0,
            'saved_files': saved_files
        }
        
        report_file = doc_base / "metadata" / f"{document_id}_{timestamp}_ocr_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(ocr_report, f, indent=2, ensure_ascii=False)
        saved_files['ocr_report'] = str(report_file)
        
        # Update tracking
        if 'visual_ocr' not in self.session_metadata['extraction_methods_used']:
            self.session_metadata['extraction_methods_used'].append('visual_ocr')
        
        self.session_metadata['total_files_created'] += len(saved_files)
        self._save_session_metadata()
        
        return saved_files
    
    def save_visual_elements_extraction(self, document_id: str, extraction_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Save visual elements extraction results with proper organization.
        
        Args:
            document_id: Document identifier  
            extraction_result: Results from visual elements extraction
            
        Returns:
            Dictionary of saved file paths
        """
        timestamp = self._get_timestamp()
        doc_base = self.base_dir / "documents" / document_id / "visual_elements"
        
        saved_files = {}
        element_manifest = []
        
        # Process extracted elements
        for element_type in ['tables', 'charts', 'images']:
            if element_type in extraction_result:
                type_dir = doc_base / "extracted" / element_type
                
                for i, element_data in enumerate(extraction_result[element_type]):
                    if 'image' in element_data:  # Has extracted image
                        element_name = f"{document_id}_{element_type[:-1]}_{i}_{timestamp}.png"
                        element_path = type_dir / element_name
                        
                        # Save image
                        element_data['image'].save(element_path)
                        saved_files[f'{element_type}_{i}_image'] = str(element_path)
                        
                        # For tables, also save CSV data if available
                        if element_type == 'tables' and 'data' in element_data:
                            csv_name = f"{document_id}_table_{i}_{timestamp}_data.csv"
                            csv_path = type_dir / csv_name
                            
                            if isinstance(element_data['data'], pd.DataFrame):
                                element_data['data'].to_csv(csv_path, index=False)
                            else:
                                # Convert dict/list to DataFrame
                                df = pd.DataFrame(element_data['data'])
                                df.to_csv(csv_path, index=False)
                            
                            saved_files[f'table_{i}_data'] = str(csv_path)
                        
                        # Add to manifest
                        element_manifest.append({
                            'type': element_type[:-1],  # Remove 's'
                            'index': i,
                            'filename': element_name,
                            'bbox': element_data.get('bbox', {}),
                            'confidence': element_data.get('confidence', 0),
                            'has_data': 'data' in element_data
                        })
        
        # Save element manifest
        manifest_file = doc_base / "metadata" / f"{document_id}_{timestamp}_elements_manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(element_manifest, f, indent=2, ensure_ascii=False)
        saved_files['elements_manifest'] = str(manifest_file)
        
        # Create extraction summary
        summary = {
            'extraction_method': 'visual_elements',
            'document_id': document_id,
            'timestamp': timestamp,
            'processing_time': extraction_result.get('processing_time', 0),
            'total_elements': len(element_manifest),
            'breakdown': {
                'tables': len([e for e in element_manifest if e['type'] == 'table']),
                'charts': len([e for e in element_manifest if e['type'] == 'chart']),
                'images': len([e for e in element_manifest if e['type'] == 'image'])
            },
            'saved_files': saved_files
        }
        
        summary_file = doc_base / "metadata" / f"{document_id}_{timestamp}_extraction_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        saved_files['extraction_summary'] = str(summary_file)
        
        # Update tracking
        if 'visual_elements' not in self.session_metadata['extraction_methods_used']:
            self.session_metadata['extraction_methods_used'].append('visual_elements')
        
        self.session_metadata['total_files_created'] += len(saved_files)
        self._save_session_metadata()
        
        return saved_files
    
    def save_enhanced_images_extraction(self, document_id: str, extraction_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Save enhanced images extraction results with comprehensive organization.
        
        Args:
            document_id: Document identifier
            extraction_result: Results from enhanced images extraction
            
        Returns:
            Dictionary of saved file paths
        """
        timestamp = self._get_timestamp()
        doc_base = self.base_dir / "documents" / document_id / "enhanced_images"
        
        saved_files = {}
        
        # Process each extracted image
        for i, img_data in enumerate(extraction_result.get('extracted_images', [])):
            img_type = img_data.get('classification', {}).get('type', 'unknown')
            visibility = img_data.get('visibility_analysis', {}).get('visibility', 'unknown')
            filename = img_data.get('filename', f'image_{i}.jpg')
            
            # Organize by type
            type_dir = doc_base / "organized" / "by_type" / img_type
            type_path = type_dir / filename
            
            # Organize by visibility  
            visibility_dir = doc_base / "organized" / "by_visibility" / visibility
            visibility_path = visibility_dir / filename
            
            # Create enhanced version
            enhanced_name = filename.replace('.jpg', f'_{timestamp}_enhanced.jpg')
            enhanced_path = doc_base / "enhanced" / enhanced_name
            
            # Copy files to organized locations (assuming source files exist)
            # This would need to be adapted based on where the original images are
            if 'source_path' in img_data and Path(img_data['source_path']).exists():
                source = Path(img_data['source_path'])
                
                # Copy to type organization
                shutil.copy2(source, type_path)
                saved_files[f'image_{i}_by_type'] = str(type_path)
                
                # Copy to visibility organization  
                shutil.copy2(source, visibility_path)
                saved_files[f'image_{i}_by_visibility'] = str(visibility_path)
                
                # Create enhanced version (placeholder - would need actual enhancement)
                shutil.copy2(source, enhanced_path)
                saved_files[f'image_{i}_enhanced'] = str(enhanced_path)
        
        # Save comprehensive metadata
        analysis_file = doc_base / "metadata" / f"{document_id}_{timestamp}_image_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_result, f, indent=2, ensure_ascii=False)
        saved_files['image_analysis'] = str(analysis_file)
        
        # Save correlation data separately for query optimization
        correlation_data = {
            'document_id': document_id,
            'timestamp': timestamp,
            'correlations': []
        }
        
        for i, img_data in enumerate(extraction_result.get('extracted_images', [])):
            if 'text_correlation' in img_data:
                correlation_data['correlations'].append({
                    'image_index': i,
                    'filename': img_data.get('filename'),
                    'correlation_strength': img_data['text_correlation'].get('correlation_strength', 0),
                    'primary_context': img_data['text_correlation'].get('primary_context'),
                    'nearby_text': img_data['text_correlation'].get('nearby_text', '')[:200]
                })
        
        correlation_file = doc_base / "metadata" / f"{document_id}_{timestamp}_correlation_data.json"
        with open(correlation_file, 'w', encoding='utf-8') as f:
            json.dump(correlation_data, f, indent=2, ensure_ascii=False)
        saved_files['correlation_data'] = str(correlation_file)
        
        # Create classification report
        classification_report = {
            'document_id': document_id,
            'timestamp': timestamp,
            'total_images': len(extraction_result.get('extracted_images', [])),
            'classification_breakdown': extraction_result.get('processing_summary', {}).get('type_distribution', {}),
            'visibility_breakdown': extraction_result.get('processing_summary', {}).get('visibility_distribution', {}),
            'average_confidence': extraction_result.get('processing_summary', {}).get('average_confidence', 0),
            'saved_files': saved_files
        }
        
        classification_file = doc_base / "metadata" / f"{document_id}_{timestamp}_classification_report.json"
        with open(classification_file, 'w', encoding='utf-8') as f:
            json.dump(classification_report, f, indent=2, ensure_ascii=False)
        saved_files['classification_report'] = str(classification_file)
        
        # Update tracking
        if 'enhanced_images' not in self.session_metadata['extraction_methods_used']:
            self.session_metadata['extraction_methods_used'].append('enhanced_images')
        
        self.session_metadata['total_files_created'] += len(saved_files)
        self._save_session_metadata()
        
        return saved_files
    
    def create_queryable_formats(self, force_refresh: bool = False) -> Dict[str, str]:
        """
        Create unified queryable formats from all processed documents.
        
        Args:
            force_refresh: Whether to recreate all queryable formats
            
        Returns:
            Dictionary of created queryable format files
        """
        queryable_dir = self.base_dir / "queryable"
        saved_files = {}
        
        # Collect all document data
        all_documents = []
        all_text_extractions = []
        all_visual_elements = []
        all_images = []
        
        documents_dir = self.base_dir / "documents"
        
        if documents_dir.exists():
            for doc_dir in documents_dir.iterdir():
                if doc_dir.is_dir():
                    doc_id = doc_dir.name
                    
                    # Load document metadata
                    doc_metadata_file = doc_dir / "document_metadata.json"
                    if doc_metadata_file.exists():
                        with open(doc_metadata_file, 'r', encoding='utf-8') as f:
                            doc_metadata = json.load(f)
                            all_documents.append(doc_metadata)
                    
                    # Collect text extractions
                    for method in ['enhanced_pdf', 'visual_ocr']:
                        method_dir = doc_dir / method / "metadata"
                        if method_dir.exists():
                            for report_file in method_dir.glob("*_extraction_report.json"):
                                with open(report_file, 'r', encoding='utf-8') as f:
                                    extraction_data = json.load(f)
                                    extraction_data['document_id'] = doc_id
                                    all_text_extractions.append(extraction_data)
                            
                            for report_file in method_dir.glob("*_ocr_report.json"):
                                with open(report_file, 'r', encoding='utf-8') as f:
                                    extraction_data = json.load(f)
                                    extraction_data['document_id'] = doc_id
                                    all_text_extractions.append(extraction_data)
                    
                    # Collect visual elements
                    elements_dir = doc_dir / "visual_elements" / "metadata"
                    if elements_dir.exists():
                        for summary_file in elements_dir.glob("*_extraction_summary.json"):
                            with open(summary_file, 'r', encoding='utf-8') as f:
                                elements_data = json.load(f)
                                elements_data['document_id'] = doc_id
                                all_visual_elements.append(elements_data)
                    
                    # Collect enhanced images
                    images_dir = doc_dir / "enhanced_images" / "metadata"
                    if images_dir.exists():
                        for analysis_file in images_dir.glob("*_image_analysis.json"):
                            with open(analysis_file, 'r', encoding='utf-8') as f:
                                images_data = json.load(f)
                                # Flatten image data for queryable format
                                for i, img in enumerate(images_data.get('extracted_images', [])):
                                    img_record = {
                                        'document_id': doc_id,
                                        'image_index': i,
                                        'filename': img.get('filename'),
                                        'type': img.get('classification', {}).get('type'),
                                        'type_confidence': img.get('classification', {}).get('confidence'),
                                        'visibility': img.get('visibility_analysis', {}).get('visibility'),
                                        'correlation_strength': img.get('text_correlation', {}).get('correlation_strength'),
                                        'context': img.get('text_correlation', {}).get('primary_context'),
                                        'description': img.get('contextual_description'),
                                        'width': img.get('dimensions', {}).get('width'),
                                        'height': img.get('dimensions', {}).get('height')
                                    }
                                    all_images.append(img_record)
        
        # Create database-ready CSV files
        if all_documents:
            documents_df = pd.DataFrame(all_documents)
            documents_csv = queryable_dir / "database_ready" / "documents_table.csv"
            documents_df.to_csv(documents_csv, index=False, encoding='utf-8')
            saved_files['documents_table'] = str(documents_csv)
        
        if all_text_extractions:
            extractions_df = pd.DataFrame(all_text_extractions)
            extractions_csv = queryable_dir / "database_ready" / "text_extractions_table.csv"
            extractions_df.to_csv(extractions_csv, index=False, encoding='utf-8')
            saved_files['text_extractions_table'] = str(extractions_csv)
        
        if all_visual_elements:
            elements_df = pd.DataFrame(all_visual_elements)
            elements_csv = queryable_dir / "database_ready" / "visual_elements_table.csv"
            elements_df.to_csv(elements_csv, index=False, encoding='utf-8')
            saved_files['visual_elements_table'] = str(elements_csv)
        
        if all_images:
            images_df = pd.DataFrame(all_images)
            images_csv = queryable_dir / "database_ready" / "images_table.csv"
            images_df.to_csv(images_csv, index=False, encoding='utf-8')
            saved_files['images_table'] = str(images_csv)
        
        # Create search indices (optimized for search engines)
        search_indices = {
            'text_search_index': [
                {
                    'id': f"text_{i}",
                    'document_id': extraction.get('document_id'),
                    'method': extraction.get('extraction_method'),
                    'content_preview': extraction.get('text_preview', ''),
                    'confidence': extraction.get('confidence_score', extraction.get('average_confidence', 0)),
                    'searchable': True
                }
                for i, extraction in enumerate(all_text_extractions)
            ],
            
            'image_search_index': [
                {
                    'id': f"img_{i}",
                    'document_id': img.get('document_id'),
                    'type': img.get('type'),
                    'description': img.get('description', ''),
                    'context': img.get('context', ''),
                    'confidence': img.get('type_confidence', 0),
                    'searchable': True
                }
                for i, img in enumerate(all_images)
            ],
            
            'elements_search_index': [
                {
                    'id': f"elem_{i}",
                    'document_id': element.get('document_id'),
                    'total_elements': element.get('total_elements', 0),
                    'breakdown': element.get('breakdown', {}),
                    'searchable': True
                }
                for i, element in enumerate(all_visual_elements)
            ]
        }
        
        # Save search indices
        for index_name, index_data in search_indices.items():
            index_file = queryable_dir / "search_indices" / f"{index_name}.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            saved_files[index_name] = str(index_file)
        
        # Create unified master index
        unified_index = {
            'created_at': datetime.now().isoformat(),
            'session_id': self.session_id,
            'total_documents': len(all_documents),
            'total_text_extractions': len(all_text_extractions),
            'total_visual_elements': sum(elem.get('total_elements', 0) for elem in all_visual_elements),
            'total_images': len(all_images),
            'available_indices': list(search_indices.keys()),
            'database_tables': [
                'documents_table.csv',
                'text_extractions_table.csv', 
                'visual_elements_table.csv',
                'images_table.csv'
            ]
        }
        
        unified_file = queryable_dir / "unified_index.json"
        with open(unified_file, 'w', encoding='utf-8') as f:
            json.dump(unified_index, f, indent=2, ensure_ascii=False)
        saved_files['unified_index'] = str(unified_file)
        
        # Create API-ready formats
        api_formats = {
            'documents_api': {
                'status': 'success',
                'total': len(all_documents),
                'data': all_documents
            },
            'extractions_api': {
                'status': 'success', 
                'total': len(all_text_extractions),
                'data': all_text_extractions
            },
            'search_api': {
                'status': 'success',
                'indices': search_indices,
                'unified_index': unified_index
            }
        }
        
        for api_name, api_data in api_formats.items():
            api_file = queryable_dir / "api_ready" / f"{api_name}.json"
            with open(api_file, 'w', encoding='utf-8') as f:
                json.dump(api_data, f, indent=2, ensure_ascii=False)
            saved_files[api_name] = str(api_file)
        
        return saved_files
    
    def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of a processed document."""
        doc_dir = self.base_dir / "documents" / document_id
        
        if not doc_dir.exists():
            return {'error': f'Document {document_id} not found'}
        
        summary = {
            'document_id': document_id,
            'processing_methods': [],
            'files_created': 0,
            'extractions': {}
        }
        
        # Check each processing method
        methods = ['enhanced_pdf', 'visual_ocr', 'visual_elements', 'enhanced_images']
        
        for method in methods:
            method_dir = doc_dir / method
            if method_dir.exists():
                summary['processing_methods'].append(method)
                
                # Count files in this method
                method_files = len(list(method_dir.rglob('*')))
                summary['files_created'] += method_files
                summary['extractions'][method] = {
                    'files_created': method_files,
                    'directories': [str(p.relative_to(method_dir)) for p in method_dir.rglob('*') if p.is_dir()]
                }
        
        return summary
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the entire session."""
        return {
            'session_metadata': self.session_metadata,
            'directory_structure': self._get_directory_tree(),
            'queryable_formats_available': self._check_queryable_formats()
        }
    
    def _get_directory_tree(self) -> Dict[str, Any]:
        """Get a tree representation of the output directory structure."""
        def build_tree(path: Path, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
            if current_depth >= max_depth:
                return {'type': 'directory', 'truncated': True}
            
            if path.is_file():
                return {
                    'type': 'file',
                    'size': path.stat().st_size,
                    'modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                }
            elif path.is_dir():
                return {
                    'type': 'directory',
                    'children': {
                        child.name: build_tree(child, max_depth, current_depth + 1)
                        for child in sorted(path.iterdir())
                        if not child.name.startswith('.')
                    }
                }
        
        return build_tree(self.base_dir)
    
    def _check_queryable_formats(self) -> Dict[str, bool]:
        """Check which queryable formats are available."""
        queryable_dir = self.base_dir / "queryable"
        
        formats = {
            'unified_index': (queryable_dir / "unified_index.json").exists(),
            'documents_table': (queryable_dir / "database_ready" / "documents_table.csv").exists(),
            'text_extractions_table': (queryable_dir / "database_ready" / "text_extractions_table.csv").exists(),
            'visual_elements_table': (queryable_dir / "database_ready" / "visual_elements_table.csv").exists(),
            'images_table': (queryable_dir / "database_ready" / "images_table.csv").exists(),
            'search_indices': len(list((queryable_dir / "search_indices").glob("*.json"))) > 0 if (queryable_dir / "search_indices").exists() else False,
            'api_formats': len(list((queryable_dir / "api_ready").glob("*.json"))) > 0 if (queryable_dir / "api_ready").exists() else False
        }
        
        return formats


def get_output_manager(session_id: Optional[str] = None) -> EcoMetricxOutputManager:
    """
    Factory function to get or create an output manager instance.
    
    Args:
        session_id: Optional session ID
        
    Returns:
        EcoMetricxOutputManager instance
    """
    return EcoMetricxOutputManager(session_id=session_id)