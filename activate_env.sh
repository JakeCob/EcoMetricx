#!/bin/bash
# Activation script for pdf-extractor environment

echo "🔄 Activating pdf-extractor conda environment..."

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate the pdf-extractor environment
conda activate pdf-extractor

echo "✅ Environment activated!"
echo "📝 Available scripts:"
echo "  python pdf_extractor.py document.pdf          # Extract text"
echo "  python image_extractor.py document.pdf        # Extract images"
echo "  python image_extractor.py document.pdf --help # See all options"
echo ""
echo "🎯 Example commands:"
echo "  python image_extractor.py task/test_info_extract.pdf --enhance --verbose"
echo "  python image_extractor.py document.pdf --min-size 200 --exclude-types logo"