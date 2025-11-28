"""
Preprocessing module for Smart Recycle Bin
Handles data splitting and organization
"""
import os
import random
import shutil
from pathlib import Path


def preprocess_data(raw_dir="/data/raw", processed_dir="/data/processed", seed=42, force=False):
    """
    Preprocess the uploaded raw data:
    1. Split into train/val/test
    2. Organize into proper directory structure
    3. Return statistics
    
    Args:
        raw_dir: Path to raw data directory
        processed_dir: Path to save processed data
        seed: Random seed for reproducibility
    
    Returns:
        dict: Status and statistics
    """
    random.seed(seed)
    
    CLASSES = ["metal", "paper", "plastic"]
    SPLIT_RATIO = (0.8, 0.1, 0.1)  # Train, Val, Test
    
    print("Starting preprocessing...")
    
    if not os.path.exists(raw_dir):
        print(f"Error: {raw_dir} not found. Please upload raw data first.")
        return {"status": "error", "message": "Raw data not found"}
    
    # Check if processed data already exists
    if os.path.exists(processed_dir) and not force:
        train_dir = os.path.join(processed_dir, "train")
        if os.path.exists(train_dir) and os.listdir(train_dir):
            print(f"Processed data already exists at {processed_dir}")
            return {"status": "skipped", "message": "Data already preprocessed. Use force=True to overwrite."}
    
    if force and os.path.exists(processed_dir):
        import shutil
        print(f"Force flag set. Removing existing data at {processed_dir}...")
        shutil.rmtree(processed_dir)
    
    # Create processed directory structure
    for split in ["train", "val", "test"]:
        for class_name in CLASSES:
            os.makedirs(os.path.join(processed_dir, split, class_name), exist_ok=True)
    
    stats = {}
    
    # Process each class
    for class_name in CLASSES:
        source_path = os.path.join(raw_dir, class_name)
        
        if not os.path.exists(source_path):
            print(f"Warning: {class_name} directory not found")
            continue
        
        # Get all image files
        images = [f for f in os.listdir(source_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        total = len(images)
        train_count = int(total * SPLIT_RATIO[0])
        val_count = int(total * SPLIT_RATIO[1])
        
        train_imgs = images[:train_count]
        val_imgs = images[train_count:train_count + val_count]
        test_imgs = images[train_count + val_count:]
        
        stats[class_name] = {
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
            "total": total
        }
        
        print(f"{class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
        
        # Copy files to respective splits
        for img_list, split in [(train_imgs, "train"), (val_imgs, "val"), (test_imgs, "test")]:
            for filename in img_list:
                src = os.path.join(source_path, filename)
                dst = os.path.join(processed_dir, split, class_name, filename)
                shutil.copy2(src, dst)
    
    print("Preprocessing complete!")
    print(f"Data saved to {processed_dir}")
    
    return {"status": "success", "stats": stats}
