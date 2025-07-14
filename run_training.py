#!/usr/bin/env python3
"""
Simple script to run the complete Siamese network training pipeline.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")

    # Map package names to their import names
    package_imports = {
        'tensorflow': 'tensorflow',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scikit-learn': 'sklearn',
        'opencv-python': 'cv2',
        'Pillow': 'PIL',
        'tqdm': 'tqdm'
    }

    missing_packages = []
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install -r requirements.txt")
        return False

    print("All dependencies are installed!")
    return True

def check_dataset():
    """Check if the dataset exists and is properly structured"""
    print("\nChecking dataset...")
    dataset_path = Path('post_processing')  # Changed from 'siamese_dataset'
    if not dataset_path.exists():
        print(f"✗ Dataset path '{dataset_path}' not found!")
        return False
    # Check if there are cat folders
    cat_folders = [f for f in dataset_path.iterdir() if f.is_dir() and f.name.startswith('cat_')]
    if not cat_folders:
        print("✗ No cat folders found in dataset!")
        return False
    print(f"✓ Found {len(cat_folders)} cat folders")
    # Check a few folders for images
    total_images = 0
    for cat_folder in cat_folders[:5]:  # Check first 5 folders
        image_files = [f for f in cat_folder.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        total_images += len(image_files)
        print(f"  {cat_folder.name}: {len(image_files)} images")
    if total_images == 0:
        print("✗ No images found in cat folders!")
        return False
    print("✓ Dataset structure looks good!")
    return True

def run_analysis():
    """Run dataset analysis"""
    print("\nRunning dataset analysis...")
    try:
        result = subprocess.run([sys.executable, 'dataset_analyzer.py', '--fast'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Dataset analysis completed successfully")
            return True
        else:
            print(f"✗ Dataset analysis failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error running dataset analysis: {e}")
        return False

def run_training():
    """Run the main training script"""
    print("\nStarting Siamese network training...")
    try:
        # Add timeout and run without capturing output for real-time feedback
        print("Running training script (this may take a while on CPU)...")
        result = subprocess.run([sys.executable, 'train_siamese.py'], 
                              timeout=3600,  # 1 hour timeout
                              text=True)
        if result.returncode == 0:
            print("✓ Training completed successfully")
            return True
        else:
            print(f"✗ Training failed with return code: {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Training timed out after 1 hour")
        return False
    except Exception as e:
        print(f"✗ Error running training: {e}")
        return False

def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description='Run Optimized Siamese Network Training Pipeline')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip dataset analysis step')
    parser.add_argument('--skip-dependency-check', action='store_true', help='Skip dependency check')
    parser.add_argument('--analysis-only', action='store_true', help='Only run dataset analysis')
    parser.add_argument('--fast', action='store_true', help='Run in fast mode (reduced parameters)')
    args = parser.parse_args()
    print("Optimized Siamese Network Training Pipeline")
    print("=" * 60)
    # Check dependencies
    if not args.skip_dependency_check:
        if not check_dependencies():
            print("\nPlease install missing dependencies and try again.")
            return 1
    # Check dataset
    if not check_dataset():
        print("\nPlease ensure your dataset is properly organized and try again.")
        return 1
    # Run analysis only if requested
    if args.analysis_only:
        if run_analysis():
            print("\nAnalysis completed successfully!")
            return 0
        else:
            print("\nAnalysis failed!")
            return 1
    # Run analysis (unless skipped)
    if not args.skip_analysis and not args.fast:
        if not run_analysis():
            print("\nDataset analysis failed! Continuing with training anyway...")
    elif args.fast:
        print("\nSkipping dataset analysis in fast mode...")
    # Run training
    if run_training():
        print("\n" + "=" * 60)
        print("Optimized Training Pipeline Completed Successfully!")
        print("\nGenerated files:")
        print("  - best_siamese_contrastive.h5 (best contrastive loss model)")
        print("  - best_siamese_triplet.h5 (best triplet loss model)")
        print("  - siamese_training_results.csv (training results)")
        print("  - training_history_contrastive.png (training plots)")
        print("  - training_history_triplet.png (training plots)")
        print("  - dataset_analysis.png (dataset statistics)")
        print("  - dataset_analysis.csv (detailed dataset info)")
        print("  - selected_cats_for_training.csv (selected cats)")
        return 0
    else:
        print("\n❌ Optimized Training Pipeline Failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 