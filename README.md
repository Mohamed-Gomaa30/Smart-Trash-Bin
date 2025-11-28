# Smart Recycle Bin - ML Pipeline

A complete Machine Learning pipeline for classifying waste (Paper, Can, Plastic) using **Modal** for remote GPU training and **TensorFlow/Keras**.

## 🚀 Quick Start

### 1. Setup
```bash
# Install dependencies
pip install modal tensorflow numpy matplotlib pillow

# Authenticate with Modal
modal setup
```

### 2. Workflow
Run all commands from the project root.

**Step 1: Upload Data**
Upload your local `data/raw` folder (containing `metal`, `paper`, `plastic` subfolders) to the Modal Volume.
```bash
modal volume put recycle-data data/raw /data
```

**Step 2: Preprocess Data**
Splits data into Train/Val/Test and saves to volume.
```bash
modal run -m modal_run::run_preprocessing
```
*Note: To overwrite existing processed data, you must manually clear the volume or modify the script to force overwrite.*

**Step 3: Train Models**
Train on remote GPUs.

*   **Transfer Learning (Recommended)**: Loads ImageNet weights and fine-tunes.
    ```bash
    modal run -m modal_run::run_training --model-name mobilenet --mode transfer
    modal run -m modal_run::run_training --model-name resnet50 --mode transfer
    ```

*   **Train from Scratch**:
    ```bash
    modal run -m modal_run::run_training --model-name mobilenet --mode scratch
    modal run -m modal_run::run_training --model-name custom_cnn
    ```

**Step 4: Download Results**
Download trained models (`.h5`) and training plots (`.png`).
```bash
# Download Models
modal volume get recycle-data /data/models models/

# Download Training Plots
modal volume get recycle-data /data/assets assets/
```

**Step 5: Local Inference**
Run real-time classification using your webcam.
```bash
python src/inference.py
```

---

## � Project Structure

```
.
├── modal_run.py        # Main entry point for Modal (Preprocessing & Training)
├── src/
│   ├── models.py       # Model definitions (MobileNet, ResNet, Custom CNN)
│   ├── train.py        # Training logic (Scratch vs Transfer, Early Stopping)
│   ├── preprocessing.py# Data splitting and processing
│   └── inference.py    # Local webcam inference script
├── data/               # Local data directory
│   ├── raw/            # Your raw images (metal, paper, plastic)
│   └── processed/      # (Optional) Local processed data
├── models/             # Downloaded trained models
└── assets/             # Downloaded training plots
```

## 🧠 Model Details

### Supported Architectures
1.  **MobileNetV2**: Lightweight, fast, ideal for edge devices.
2.  **ResNet50**: Deeper network, higher accuracy, heavier.
3.  **VGG16**: Classic deep network.
4.  **Custom CNN**: Simple CNN trained from scratch.

### Training Modes
*   **Transfer (`--mode transfer`)**:
    *   Loads **ImageNet** weights.
    *   Freezes bottom layers (feature extractors).
    *   Unfreezes top layers for fine-tuning.
    *   Uses `EarlyStopping` (patience=10) and `Dropout` (0.4) to prevent overfitting.
*   **Scratch (`--mode scratch`)**:
    *   Initializes with random weights.
    *   Trains all layers from epoch 0.

## �️ Advanced Usage

### Expanding the Dataset
To add more data (e.g., merging **TrashNet** or **RealWaste**):
1.  Manually add images to `data/raw/metal`, `data/raw/paper`, etc.
2.  Re-upload to Modal: `modal volume put recycle-data data/raw /data`
3.  Re-run preprocessing: `modal run -m modal_run::run_preprocessing`

### Volume Management
Inspect volume contents:
```bash
modal volume ls recycle-data
```
Delete a file/folder:
```bash
modal volume rm recycle-data /data/path/to/file
```
