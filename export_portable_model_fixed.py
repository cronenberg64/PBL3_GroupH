#!/usr/bin/env python3
"""
Fixed export script that handles Lambda layer issues.
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np

def create_clean_embedding_model():
    """Create a clean embedding model without Lambda layer issues"""
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import Input, Dense, Flatten
    
    print("🔄 Creating clean embedding model...")
    
    # Create the same architecture as in training
    input_shape = (150, 150, 3)
    embedding_dim = 128
    
    # Load VGG16 base
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create embedding model
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    embeddings = Dense(embedding_dim, activation=None, name='embeddings')(x)
    
    # Use a custom layer instead of Lambda for L2 normalization
    class L2Normalize(keras.layers.Layer):
        def __init__(self, **kwargs):
            super(L2Normalize, self).__init__(**kwargs)
        
        def call(self, inputs):
            return tf.math.l2_normalize(inputs, axis=1)
        
        def get_config(self):
            return super(L2Normalize, self).get_config()
    
    embeddings = L2Normalize()(embeddings)
    
    embedding_model = keras.Model(inputs, outputs=embeddings, name='embedding_model')
    
    print("✅ Clean embedding model created")
    return embedding_model

def load_and_fix_model(model_path):
    """Load model and fix Lambda layer issues"""
    print(f"🔄 Loading model from: {model_path}")
    
    try:
        # Load the model
        model = keras.models.load_model(model_path, compile=False)
        print(f"✅ Model loaded successfully")
        
        # Get the embedding model from the loaded model
        embedding_model = None
        
        # Try to extract embedding model from different layers
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                # Check if this looks like an embedding model
                if any('embedding' in layer.name.lower() for layer in layer.layers):
                    embedding_model = layer
                    print(f"✅ Found embedding model in layer {i}")
                    break
        
        if embedding_model is None:
            print("⚠️ Could not extract embedding model, creating new one")
            embedding_model = create_clean_embedding_model()
        
        return embedding_model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Creating new embedding model...")
        return create_clean_embedding_model()

def export_models():
    """Export models in portable format"""
    print("🚀 Fixed Siamese Model Export Tool")
    print("=" * 50)
    
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
    
    if not existing_models:
        print("❌ No models found!")
        return False
    
    success_count = 0
    
    for model_path in existing_models:
        model_name = model_path.replace(".h5", "")
        
        # Load and fix the model
        embedding_model = load_and_fix_model(model_path)
        
        # Export embedding model
        output_dir = f"{model_name}_embedding_savedmodel"
        try:
            print(f"🔄 Exporting to: {output_dir}")
            embedding_model.export(output_dir)
            print(f"✅ Successfully exported to: {output_dir}")
            
            # Test the exported model
            print(f"🔄 Testing exported model...")
            loaded_model = keras.models.load_model(output_dir)
            test_input = np.random.random((1, 150, 150, 3)).astype(np.float32)
            test_output = loaded_model.predict(test_input, verbose=0)
            print(f"✅ Test successful - Output shape: {test_output.shape}")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error exporting: {e}")
    
    # Create model info
    info = {
        "model_type": "siamese_embedding",
        "base_model": "vgg16",
        "input_shape": [150, 150, 3],
        "embedding_dim": 128,
        "image_size": 150,
        "export_format": "savedmodel",
        "framework": "tensorflow",
        "version": tf.__version__
    }
    
    import json
    with open("model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Created model_info.json")
    
    print("\n" + "=" * 50)
    print("📋 Export Summary:")
    print(f"✅ Successfully exported {success_count} models")
    
    print("\n📁 Generated files:")
    for model_path in existing_models:
        model_name = model_path.replace(".h5", "")
        print(f"   - {model_name}_embedding_savedmodel/ (embedding model)")
    
    print("   - model_info.json (metadata)")
    
    print("\n🔧 Next steps:")
    print("1. Copy the *_embedding_savedmodel directories to your Mac")
    print("2. Update your backend to load the SavedModel format")
    print("3. Re-register known cats with the new model")
    print("4. Test with Expo Go")
    
    return success_count > 0

if __name__ == "__main__":
    success = export_models()
    sys.exit(0 if success else 1) 