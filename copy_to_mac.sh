#!/bin/bash
# Script to copy exported models to Mac

echo "🚀 Model Transfer Script"
echo "========================"

# Configuration - UPDATE THESE VALUES
MAC_USERNAME="jonathansetiawan"
MAC_IP="192.168.40.138"
MAC_PATH="/Users/jonathansetiawan/Documents/Programming_Projects/PBL3_GroupH/"

echo "📋 Configuration:"
echo "   Mac Username: $MAC_USERNAME"
echo "   Mac IP: $MAC_IP"
echo "   Mac Path: $MAC_PATH"
echo ""

# Check if exported models exist
MODELS_TO_COPY=(
    "best_siamese_contrastive_savedmodel"
    "best_siamese_contrastive_embedding_savedmodel"
    "best_siamese_triplet_savedmodel"
    "best_siamese_triplet_embedding_savedmodel"
    "model_info.json"
)

echo "🔍 Checking for exported models..."
for model in "${MODELS_TO_COPY[@]}"; do
    if [ -e "$model" ]; then
        echo "✅ Found: $model"
    else
        echo "❌ Missing: $model"
    fi
done

echo ""
echo "🔄 Starting transfer..."

# Copy each model
for model in "${MODELS_TO_COPY[@]}"; do
    if [ -e "$model" ]; then
        echo "📤 Copying $model..."
        scp -r "$model" "$MAC_USERNAME@$MAC_IP:$MAC_PATH"
        if [ $? -eq 0 ]; then
            echo "✅ Successfully copied $model"
        else
            echo "❌ Failed to copy $model"
        fi
    fi
done

echo ""
echo "🎉 Transfer complete!"
echo ""
echo "📋 Next steps on your Mac:"
echo "1. Update your backend to load SavedModel format"
echo "2. Re-register known cats: python ai_model/register_known_cats.py"
echo "3. Restart your backend server"
echo "4. Test with Expo Go" 