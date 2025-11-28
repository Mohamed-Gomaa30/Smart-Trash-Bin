"""
Training module for Smart Recycle Bin
Handles model training with flexible modes (scratch vs transfer)
"""
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

from src.models import get_model, get_fine_tune_layer


def train_model(
    model_name,
    mode='transfer', 
    data_dir="/data/processed",
    model_dir="/data/models",
    assets_dir="/data/assets",
    results_dir="/data/results",
    img_size=(224, 224),
    batch_size=32,
    epochs=15,
    fine_tune_epochs=5,
    num_classes=3
):
    """
    Train a model
    """
    print("=" * 60)
    print(f"Training {model_name.upper()} (Mode: {mode})")
    print("=" * 60)
    
    # GPU Info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus[0]}")
    else:
        print("No GPU detected - using CPU")
    print()
    
    # Check data
    if not os.path.exists(os.path.join(data_dir, "train")):
        return {"status": "error", "message": "Processed data not found."}
    
    # Load datasets
    train_ds = image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    val_ds = image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=img_size,
        batch_size=batch_size,
        seed=42
    )
    
    test_ds = image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        image_size=img_size,
        batch_size=batch_size,
        seed=42
    )
    
    # Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.3), 
        tf.keras.layers.RandomZoom(0.3), 
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])
    
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Create Model
    model = get_model(model_name, input_shape=img_size + (3,), num_classes=num_classes, mode=mode)
    
    # Configure Training Mode
    if mode == 'scratch' or model_name == 'custom_cnn':
        print(f"Initializing {model_name} from scratch (Random Weights)...")
        # All layers are trainable by default
        lr = 0.001
    else:
        print(f"Loading {model_name} with ImageNet weights for Fine-Tuning...")
        # Transfer Learning / Fine-Tuning Logic
        # We want to train the head AND fine-tune the top layers directly
        
        # 1. Base model is loaded with weights='imagenet' in get_model
        # 2. We need to unfreeze the top layers
        if hasattr(model, 'layers') and len(model.layers) > 1:
            base_model = model.layers[1] # Assuming base model is the second layer (after Input)
            base_model.trainable = True
            
            # Freeze bottom layers
            fine_tune_at = get_fine_tune_layer(model_name)
            if fine_tune_at > 0:
                print(f"Freezing bottom {fine_tune_at} layers...")
                for layer in base_model.layers[:fine_tune_at]:
                    layer.trainable = False
            else:
                print("Warning: Fine-tune layer index is 0, training all layers.")
        
        lr = 0.00005 # Lower learning rate (5e-5) for fine-tuning

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train (Single Phase)
    current_epochs = epochs if (mode == 'scratch' or model_name == 'custom_cnn') else fine_tune_epochs
    print(f"Starting training for {current_epochs} epochs...")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=current_epochs,
        verbose=1
    )
    final_history = history

    # Evaluation
    print("\nEvaluating on Test Set...")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Detailed Metrics (Confusion Matrix & Classification Report)
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    # Get predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Class names (assuming alphabetical order from image_dataset_from_directory)
    class_names = sorted(os.listdir(os.path.join(data_dir, "train")))
    
    # Setup Output Directories
    model_assets_dir = os.path.join(assets_dir, model_name)
    model_results_dir = os.path.join(results_dir, model_name)
    
    os.makedirs(model_assets_dir, exist_ok=True)
    os.makedirs(model_results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save Report
    report_path = os.path.join(model_results_dir, f"{mode}_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
        
    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name.upper()} ({mode})')
    
    cm_path = os.path.join(model_assets_dir, f"{mode}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix saved to {cm_path}")
    
    # Save Model
    model_filename = f"{model_name}_{mode}.h5"
    model_path = os.path.join(model_dir, model_filename)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save Plots
    plot_filename = f"{mode}_history.png"
    plot_path = os.path.join(model_assets_dir, plot_filename)
    
    plot_history(final_history, plot_path, title=f"{model_name.upper()} ({mode})")
    print(f"Plot saved to {plot_path}")
    
    return {
        "status": "success",
        "model_name": model_name,
        "mode": mode,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "model_path": model_path,
        "plot_path": plot_path,
        "report_path": report_path,
        "confusion_matrix_path": cm_path
    }


def plot_history(history, save_path, title="Training History"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{title} - Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{title} - Loss')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
