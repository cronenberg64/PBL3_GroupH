import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

def analyze_dataset(dataset_path):
    """Analyze the structure and statistics of the dataset"""
    print("Analyzing dataset structure...")
    
    # Get all cat folders
    cat_folders = [f for f in os.listdir(dataset_path) 
                  if f.startswith('cat_') and os.path.isdir(os.path.join(dataset_path, f))]
    cat_folders.sort()
    
    print(f"Found {len(cat_folders)} cat folders")
    
    # Analyze each cat folder
    cat_stats = []
    total_images = 0
    image_extensions = Counter()
    
    for cat_folder in cat_folders:
        cat_path = os.path.join(dataset_path, cat_folder)
        
        # Count images by extension
        image_files = [f for f in os.listdir(cat_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        extensions = [os.path.splitext(f)[1].lower() for f in image_files]
        image_extensions.update(extensions)
        
        # Check if info.json exists
        info_file = os.path.join(cat_path, 'info.json')
        has_info = os.path.exists(info_file)
        
        cat_stats.append({
            'cat_id': cat_folder,
            'num_images': len(image_files),
            'has_info': has_info,
            'extensions': list(set(extensions))
        })
        
        total_images += len(image_files)
    
    # Create summary
    summary = {
        'total_cats': len(cat_folders),
        'total_images': total_images,
        'avg_images_per_cat': total_images / len(cat_folders),
        'image_extensions': dict(image_extensions),
        'cats_with_info': sum(1 for cat in cat_stats if cat['has_info']),
        'cats_without_info': sum(1 for cat in cat_stats if not cat['has_info'])
    }
    
    return cat_stats, summary

def visualize_dataset_stats(cat_stats, summary):
    """Create visualizations of dataset statistics"""
    print("Creating dataset visualizations...")
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(cat_stats)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution of images per cat
    ax1.hist(df['num_images'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Number of Images per Cat')
    ax1.set_ylabel('Number of Cats')
    ax1.set_title('Distribution of Images per Cat')
    ax1.grid(True, alpha=0.3)
    
    # 2. Image count by cat (top 20)
    top_cats = df.nlargest(20, 'num_images')
    ax2.barh(range(len(top_cats)), top_cats['num_images'], color='lightcoral')
    ax2.set_yticks(range(len(top_cats)))
    ax2.set_yticklabels([cat_id[:15] + '...' if len(cat_id) > 15 else cat_id 
                        for cat_id in top_cats['cat_id']])
    ax2.set_xlabel('Number of Images')
    ax2.set_title('Top 20 Cats by Image Count')
    ax2.grid(True, alpha=0.3)
    
    # 3. Image extensions distribution
    extensions = summary['image_extensions']
    ax3.pie(extensions.values(), labels=extensions.keys(), autopct='%1.1f%%')
    ax3.set_title('Distribution of Image File Types')
    
    # 4. Info.json availability
    info_counts = [summary['cats_with_info'], summary['cats_without_info']]
    info_labels = ['With Info', 'Without Info']
    ax4.bar(info_labels, info_counts, color=['lightgreen', 'lightcoral'])
    ax4.set_ylabel('Number of Cats')
    ax4.set_title('Info.json Availability')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def create_training_subset(cat_stats, max_cats=20, min_images=5):
    """Create a subset of cats suitable for training"""
    print(f"Creating training subset with max {max_cats} cats and min {min_images} images per cat...")
    
    # Filter cats with minimum images
    suitable_cats = [cat for cat in cat_stats if cat['num_images'] >= min_images]
    
    # Sort by number of images (descending) and take top max_cats
    suitable_cats.sort(key=lambda x: x['num_images'], reverse=True)
    selected_cats = suitable_cats[:max_cats]
    
    print(f"Selected {len(selected_cats)} cats for training:")
    for cat in selected_cats:
        print(f"  {cat['cat_id']}: {cat['num_images']} images")
    
    return selected_cats

def save_analysis_results(cat_stats, summary, selected_cats):
    """Save analysis results to files"""
    print("Saving analysis results...")
    
    # Save detailed cat statistics
    df = pd.DataFrame(cat_stats)
    df.to_csv('dataset_analysis.csv', index=False)
    
    # Save summary
    with open('dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save selected cats for training
    selected_df = pd.DataFrame(selected_cats)
    selected_df.to_csv('selected_cats_for_training.csv', index=False)
    
    print("Analysis results saved to:")
    print("  - dataset_analysis.csv")
    print("  - dataset_summary.json")
    print("  - selected_cats_for_training.csv")
    print("  - dataset_analysis.png")

def main():
    """Main analysis function"""
    dataset_path = 'siamese_dataset'
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found!")
        return
    
    # Analyze dataset
    cat_stats, summary = analyze_dataset(dataset_path)
    
    # Print summary
    print("\nDataset Summary:")
    print("=" * 50)
    print(f"Total cats: {summary['total_cats']}")
    print(f"Total images: {summary['total_images']}")
    print(f"Average images per cat: {summary['avg_images_per_cat']:.1f}")
    print(f"Image extensions: {summary['image_extensions']}")
    print(f"Cats with info.json: {summary['cats_with_info']}")
    print(f"Cats without info.json: {summary['cats_without_info']}")
    
    # Visualize statistics
    df = visualize_dataset_stats(cat_stats, summary)
    
    # Create training subset
    selected_cats = create_training_subset(cat_stats, max_cats=20, min_images=5)
    
    # Save results
    save_analysis_results(cat_stats, summary, selected_cats)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 