#!/bin/bash
# Activation script for pdf-extractor environment

echo "🔄 Activating pdf-extractor conda environment..."

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate the pdf-extractor environment
conda activate pdf-extractor

echo "✅ Environment activated!"
echo "📝 Available scripts:"
echo "  python enhanced_pdf_extractor.py document.pdf        # Enhanced text extraction"
echo "  python enhanced_image_extractor.py document.pdf     # Enhanced image extraction"
echo "  python integrated_extractor.py document.pdf         # Full integrated processing"
echo "  python visual_pdf_extractor.py document.pdf --method hybrid  # Visual OCR extraction"
echo ""
echo "🎯 Example commands:"
echo "  python visual_pdf_extractor.py task/test_info_extract.pdf --method hybrid"
echo "  python visual_pdf_extractor.py document.pdf --method screenshot --preprocess"
echo "  python integrated_extractor.py document.pdf --full-analysis"