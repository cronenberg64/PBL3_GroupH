#!/bin/bash

# Setup script for campus workbench deployment
# Run this script after cloning the repository

echo "🚀 Setting up Cat Re-identification System for Campus Workbench"
echo "================================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Check GPU availability
echo "🔍 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️ No NVIDIA GPU detected. Training will be slower on CPU."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p post_processing
mkdir -p results
mkdir -p models

# Set production mode
echo "⚙️ Configuring for production mode..."
if [ -f "train_siamese.py" ]; then
    # Backup original file
    cp train_siamese.py train_siamese.py.backup
    
    # Replace DEBUG_MODE = True with DEBUG_MODE = False
    sed -i 's/DEBUG_MODE = True/DEBUG_MODE = False/' train_siamese.py
    
    echo "✅ Production mode enabled (DEBUG_MODE = False)"
else
    echo "⚠️ train_siamese.py not found. Please set DEBUG_MODE = False manually."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Download your dataset to post_processing/ directory"
echo "2. Run: python run_training.py"
echo "3. Monitor training progress"
echo ""
echo "📊 Expected training time: 30-60 minutes on GPU"
echo "📁 Results will be saved in: results/ directory"
echo ""
echo "🔧 To revert to debug mode:"
echo "   sed -i 's/DEBUG_MODE = False/DEBUG_MODE = True/' train_siamese.py" 