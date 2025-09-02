echo "🚀 Setting up PDF Extractor Environment"
echo "======================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment..."
conda env create -f environment.yml

# Activate environment
echo "🔄 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate pdf-extractor

# Verify installation
echo "🔍 Verifying installation..."
python -c "import pymupdf4llm; print('✅ pymupdf4llm installed successfully')"
python -c "import pymupdf; print('✅ PyMuPDF installed successfully')"
python -c "import PIL; print('✅ Pillow installed successfully')"
python -c "import pytesseract; print('✅ pytesseract installed successfully')"

# Run tests
echo "🧪 Running tests..."
python test_extraction.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "   conda activate pdf-extractor"
echo ""
echo "To test with your PDF, run:"
echo "   python pdf_extractor.py your_document.pdf"
echo ""
echo "To extract images from your PDF, run:"
echo "   python image_extractor.py your_document.pdf"
echo "   python image_extractor.py your_document.pdf --enhance --verbose"