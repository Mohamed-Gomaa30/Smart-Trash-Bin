"""
Modal Run - Entry point for Smart Recycle Bin ML Pipeline

Usage:
    # Preprocess data
    python modal_run.py preprocess
    # To run directly with Modal:
    # modal run -m modal_run::run_preprocessing --force
    # modal run -m modal_run::run_training --model-name mobilenet --mode scratch
        
    # Train a specific model
    python modal_run.py train --model mobilenet
    python modal_run.py train --model resnet50
    python modal_run.py train --model custom_cnn
    
    # Train all models
    python modal_run.py train --model all
"""

import os
import modal
from pathlib import Path 

current_dir = Path(__file__).parent
config_data = current_dir / "data/raw"
config_src = current_dir / "src"
# Create Modal app
app = modal.App("recycle-bin")

# Create or reference volume
volume = modal.Volume.from_name("recycle-data", create_if_missing=True)
print(f"Volume out {volume}")

# Define image with dependencies
image = (modal.Image.debian_slim().pip_install(
    "tensorflow[and-cuda]",
    "numpy",
    "matplotlib",
    "pillow",
    "scikit-learn",
    "seaborn"
)
.add_local_dir(config_src, remote_path="/root/src/")
.add_local_dir(config_data, remote_path="/root/data/raw/")
)


@app.function(
    image=image,
    volumes={"/data/": volume},
    timeout=3600
)
def run_preprocessing():
    """Run preprocessing on Modal"""
    import sys
    sys.path.insert(0, '/root/')
    from src.preprocessing import preprocess_data
    
    print("=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)
    print()
    
    import os
    print(f"Listing /data/: {os.listdir('/data/') if os.path.exists('/data/') else 'Not found'}")

    result = preprocess_data(
        raw_dir="data/raw", 
        processed_dir="/data/processed",
        seed=42
    )

    print()
    print("=" * 60)
    print("PREPROCESSING RESULTS")
    print("=" * 60)
    
    if result["status"] == "success":
        print("✓ Preprocessing completed successfully!")
        print()
        print("Dataset Statistics:")
        for class_name, stats in result["stats"].items():
            print(f"  {class_name.capitalize()}:")
            print(f"    Train: {stats['train']}")
            print(f"    Val:   {stats['val']}")
            print(f"    Test:  {stats['test']}")
            print(f"    Total: {stats['total']}")
    
    # Commit changes to volume
    volume.commit()

@app.function(
    image=image,
    # gpu="t4",
    volumes={"/data/": volume},
    timeout=7200
)
def run_training(model_name: str, mode: str = "transfer"):
    """Run training on Modal with GPU"""
    from src.train import train_model
    
    result = train_model(
        model_name=model_name,
        mode=mode,
        data_dir="/data/processed",
        model_dir="/data/models",
        img_size=(224, 224),
        batch_size=32,
        epochs=30,
        fine_tune_epochs=30,
        num_classes=3
    )
    
    # Commit changes to volume
    volume.commit()
    
    return result