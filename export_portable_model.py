#!/usr/bin/env python3
"""
Export trained Siamese model in portable format for cross-platform use.
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np

def check_model_exists():
    """Check if the trained model exists"""
    model_files = [
        "best_siamese_contrastive.h5",
        "best_siamese_triplet.h5"
    ]
    
    existing_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            existing_models.append(model_file)
            print(f"✅ Found: {model_file}")
        else:
            print(f"❌ Missing: {model_file}")
    
    return existing_models

def export_saved_model(model_path, output_dir):
    """Export model as SavedModel format"""
    print(f"\n🔄 Loading model from: {model_path}")
    
    try:
        # Load the model without compilation (for inference)
        model = keras.models.load_model(model_path, compile=False)
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model)}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export as SavedModel (portable across OS and Python versions)
        print(f"🔄 Exporting to: {output_dir}")
        model.save(output_dir, save_format="tf")
        
        print(f"✅ Successfully exported to: {output_dir}")
        
        # Verify the export
        print(f"🔄 Verifying export...")
        loaded_model = keras.models.load_model(output_dir)
        print(f"✅ Export verification successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting model: {e}")
        return False

def export_embedding_model(model_path, output_dir):
    """Extract and export just the embedding model"""
    print(f"\n🔄 Extracting embedding model from: {model_path}")
    
    try:
        # Load the full model
        model = keras.models.load_model(model_path, compile=False)
        
        # Try to extract the embedding model (layer 2 for contrastive, layer 3 for triplet)
        embedding_model = None
        
        # Check if it's a contrastive model (has distance layer)
        if len(model.layers) >= 3:
            # For contrastive model, the embedding model is typically layer 2
            potential_embedding = model.layers[2]
            if hasattr(potential_embedding, 'layers') and len(potential_embedding.layers) > 0:
                embedding_model = potential_embedding
                print(f"✅ Found embedding model in layer 2")
        
        # If not found, try layer 3 (for triplet models)
        if embedding_model is None and len(model.layers) >= 4:
            potential_embedding = model.layers[3]
            if hasattr(potential_embedding, 'layers') and len(potential_embedding.layers) > 0:
                embedding_model = potential_embedding
                print(f"✅ Found embedding model in layer 3")
        
        if embedding_model is None:
            print(f"⚠️ Could not extract embedding model, using full model")
            embedding_model = model
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export embedding model
        print(f"🔄 Exporting embedding model to: {output_dir}")
        embedding_model.save(output_dir, save_format="tf")
        
        print(f"✅ Successfully exported embedding model to: {output_dir}")
        
        # Test the embedding model
        print(f"🔄 Testing embedding model...")
        test_input = np.random.random((1, 150, 150, 3)).astype(np.float32)
        test_output = embedding_model.predict(test_input, verbose=0)
        print(f"✅ Test successful - Input shape: {test_input.shape}, Output shape: {test_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting embedding model: {e}")
        return False

def create_model_info():
    """Create a model info file with metadata"""
    info = {
        "model_type": "siamese_contrastive",
        "base_model": "vgg16",
        "input_shape": [150, 150, 3],
        "embedding_dim": 128,
        "image_size": 150,
        "margin": 1.0,
        "threshold": 0.4,
        "export_format": "savedmodel",
        "framework": "tensorflow",
        "version": tf.__version__
    }
    
    import json
    with open("model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Created model_info.json with metadata")

def main():
    """Main export function"""
    print("🚀 Siamese Model Export Tool")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for existing models
    existing_models = check_model_exists()
    
    if not existing_models:
        print("\n❌ No trained models found!")
        print("Please run training first: python run_training.py")
        return False
    
    # Export each model
    success_count = 0
    
    for model_path in existing_models:
        model_name = model_path.replace(".h5", "")
        
        # Export full model
        full_model_dir = f"{model_name}_savedmodel"
        if export_saved_model(model_path, full_model_dir):
            success_count += 1
        
        # Export embedding model
        embedding_dir = f"{model_name}_embedding_savedmodel"
        if export_embedding_model(model_path, embedding_dir):
            success_count += 1
    
    # Create model info
    create_model_info()
    
    print("\n" + "=" * 50)
    print("📋 Export Summary:")
    print(f"✅ Successfully exported {success_count} models")
    print("\n📁 Generated files:")
    
    # List generated files
    for model_path in existing_models:
        model_name = model_path.replace(".h5", "")
        print(f"   - {model_name}_savedmodel/ (full model)")
        print(f"   - {model_name}_embedding_savedmodel/ (embedding only)")
    
    print("   - model_info.json (metadata)")
    
    print("\n🔧 Next steps:")
    print("1. Copy the *_savedmodel directories to your Mac")
    print("2. Update your backend to load the SavedModel format")
    print("3. Re-register known cats with the new model")
    print("4. Test with Expo Go")
    
    return success_count > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 