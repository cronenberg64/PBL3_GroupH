#!/usr/bin/env python3
"""
Production Deployment Script for Cat Re-identification System
"""

import os
import sys
import json
import shutil
from pathlib import Path
import subprocess
import argparse

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking production dependencies...")
    
    required_packages = [
        'tensorflow>=2.16.0',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'Pillow>=10.0.0'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.split('>=')[0].replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        for package in missing_packages:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package])
    
    print("All dependencies are installed!")

def create_production_structure():
    """Create production directory structure"""
    print("Creating production directory structure...")
    
    production_dirs = [
        'production',
        'production/models',
        'production/database',
        'production/logs',
        'production/config',
        'production/examples'
    ]
    
    for dir_path in production_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")

def copy_model_files():
    """Copy trained models to production"""
    print("Copying model files to production...")
    
    model_files = [
        'best_siamese_contrastive.h5',
        'best_siamese_triplet.h5'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            shutil.copy2(model_file, f'production/models/{model_file}')
            print(f"✓ Copied {model_file}")
        else:
            print(f"⚠ {model_file} not found")

def create_production_config():
    """Create production configuration file"""
    print("Creating production configuration...")
    
    config = {
        "model": {
            "default_model": "best_siamese_contrastive.h5",
            "embedding_dim": 256,
            "img_size": 200,
            "threshold": 0.5
        },
        "database": {
            "path": "production/database/cat_database.pkl",
            "backup_path": "production/database/backup/"
        },
        "logging": {
            "level": "INFO",
            "file": "production/logs/app.log",
            "max_size": "10MB",
            "backup_count": 5
        },
        "performance": {
            "batch_size": 1,
            "max_workers": 4,
            "timeout": 30
        }
    }
    
    with open('production/config/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Created production configuration")

def create_production_app():
    """Create production-ready application"""
    print("Creating production application...")
    
    app_code = '''#!/usr/bin/env python3
"""
Production Cat Re-identification Application
"""

import os
import sys
import json
import logging
from pathlib import Path
from inference import CatReidentifier
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production/logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ProductionCatReidentifier:
    def __init__(self, config_path="production/config/config.json"):
        """Initialize production cat re-identifier"""
        self.config = self._load_config(config_path)
        self.reidentifier = None
        self._initialize_reidentifier()
    
    def _load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _initialize_reidentifier(self):
        """Initialize the re-identifier"""
        try:
            model_path = f"production/models/{self.config['model']['default_model']}"
            self.reidentifier = CatReidentifier(
                model_path=model_path,
                embedding_dim=self.config['model']['embedding_dim'],
                img_size=self.config['model']['img_size']
            )
            
            # Load existing database if available
            db_path = self.config['database']['path']
            if os.path.exists(db_path):
                self.reidentifier.load_database(db_path)
                logger.info(f"Loaded database with {len(self.reidentifier.known_cats)} cats")
            
            logger.info("Re-identifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing re-identifier: {e}")
    
    def add_cat(self, cat_id: str, image_path: str) -> bool:
        """Add a cat to the database"""
        try:
            success = self.reidentifier.add_known_cat(cat_id, image_path)
            if success:
                self.reidentifier.save_database(self.config['database']['path'])
                logger.info(f"Added cat {cat_id} to database")
            return success
        except Exception as e:
            logger.error(f"Error adding cat {cat_id}: {e}")
            return False
    
    def identify_cat(self, image_path: str):
        """Identify a cat from an image"""
        try:
            cat_id, confidence = self.reidentifier.identify_cat(image_path)
            logger.info(f"Identification result: {cat_id} (confidence: {confidence:.3f})")
            return cat_id, confidence
        except Exception as e:
            logger.error(f"Error identifying cat: {e}")
            return None, 0.0
    
    def get_database_info(self):
        """Get database information"""
        return self.reidentifier.get_database_info()

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Production Cat Re-identification')
    parser.add_argument('--action', choices=['add', 'identify', 'info'], required=True,
                       help='Action to perform')
    parser.add_argument('--cat-id', help='Cat ID for add action')
    parser.add_argument('--image', required=True, help='Image path')
    parser.add_argument('--config', default='production/config/config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize application
    app = ProductionCatReidentifier(args.config)
    
    if args.action == 'add':
        if not args.cat_id:
            print("Error: --cat-id is required for add action")
            return
        
        success = app.add_cat(args.cat_id, args.image)
        if success:
            print(f"✅ Successfully added cat {args.cat_id}")
        else:
            print(f"❌ Failed to add cat {args.cat_id}")
    
    elif args.action == 'identify':
        cat_id, confidence = app.identify_cat(args.image)
        if cat_id:
            print(f"✅ Identified as: {cat_id} (confidence: {confidence:.3f})")
        else:
            print(f"❌ No match found (confidence: {confidence:.3f})")
    
    elif args.action == 'info':
        info = app.get_database_info()
        print(f"Database contains {info['num_cats']} cats")
        print(f"Cat IDs: {info['cat_ids']}")

if __name__ == "__main__":
    main()
'''
    
    with open('production/app.py', 'w') as f:
        f.write(app_code)
    
    # Make it executable
    os.chmod('production/app.py', 0o755)
    print("✓ Created production application")

def create_dockerfile():
    """Create Dockerfile for containerization"""
    print("Creating Dockerfile...")
    
    dockerfile_content = '''FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY production/ ./production/
COPY inference.py .

# Create necessary directories
RUN mkdir -p production/logs production/database

# Expose port (if needed for web interface)
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "production/app.py", "--help"]
'''
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    
    print("✓ Created Dockerfile")

def create_requirements():
    """Create requirements.txt for production"""
    print("Creating requirements.txt...")
    
    requirements = '''tensorflow>=2.16.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
tqdm>=4.65.0
'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✓ Created requirements.txt")

def create_readme():
    """Create production README"""
    print("Creating production README...")
    
    readme_content = '''# Cat Re-identification System - Production

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run deployment script
python deploy.py
```

### 2. Add Cats to Database
```bash
python production/app.py --action add --cat-id "fluffy_001" --image "path/to/cat/image.jpg"
```

### 3. Identify Cats
```bash
python production/app.py --action identify --image "path/to/unknown/cat.jpg"
```

### 4. Check Database Info
```bash
python production/app.py --action info
```

## Configuration

Edit `production/config/config.json` to customize:
- Model settings (embedding dimension, image size, threshold)
- Database paths
- Logging configuration
- Performance settings

## Docker Deployment

```bash
# Build image
docker build -t cat-reidentifier .

# Run container
docker run -v $(pwd)/production:/app/production cat-reidentifier
```

## API Usage

```python
from production.app import ProductionCatReidentifier

# Initialize
app = ProductionCatReidentifier()

# Add cat
app.add_cat("fluffy_001", "path/to/image.jpg")

# Identify cat
cat_id, confidence = app.identify_cat("path/to/unknown.jpg")
```

## Monitoring

- Logs: `production/logs/app.log`
- Database: `production/database/cat_database.pkl`
- Configuration: `production/config/config.json`
'''
    
    with open('production/README.md', 'w') as f:
        f.write(readme_content)
    
    print("✓ Created production README")

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy Cat Re-identification System')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency check')
    parser.add_argument('--docker', action='store_true', help='Create Docker files')
    
    args = parser.parse_args()
    
    print("🚀 Deploying Cat Re-identification System to Production")
    print("=" * 60)
    
    # Check dependencies
    if not args.skip_deps:
        check_dependencies()
    
    # Create production structure
    create_production_structure()
    
    # Copy model files
    copy_model_files()
    
    # Create configuration
    create_production_config()
    
    # Create production application
    create_production_app()
    
    # Create requirements
    create_requirements()
    
    # Create README
    create_readme()
    
    # Create Docker files if requested
    if args.docker:
        create_dockerfile()
    
    print("\n" + "=" * 60)
    print("✅ Production Deployment Complete!")
    print("\nNext steps:")
    print("1. Test the system: python production/app.py --action info")
    print("2. Add cats: python production/app.py --action add --cat-id 'test_cat' --image 'path/to/image.jpg'")
    print("3. Identify cats: python production/app.py --action identify --image 'path/to/image.jpg'")
    print("4. Check logs: tail -f production/logs/app.log")
    
    if args.docker:
        print("\nDocker commands:")
        print("docker build -t cat-reidentifier .")
        print("docker run -v $(pwd)/production:/app/production cat-reidentifier")

if __name__ == "__main__":
    main() 