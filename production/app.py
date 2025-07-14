#!/usr/bin/env python3
"""
Production Cat Re-identification Application
"""

import os
import sys
import json
import logging
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
