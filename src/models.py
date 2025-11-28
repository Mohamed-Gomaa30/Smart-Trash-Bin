"""
Model definitions for Smart Recycle Bin
Supports multiple architectures: MobileNetV2, ResNet50, VGG16, Custom CNN
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential


def create_transfer_model(model_name, input_shape=(224, 224, 3), num_classes=3, weights='imagenet', trainable_base=False):
    """
    Create a model based on a pre-trained architecture
    
    Args:
        model_name: Name of the architecture ('mobilenet', 'resnet50', 'vgg16')
        input_shape: Input image shape
        num_classes: Number of output classes
        weights: 'imagenet' for transfer learning, or None for training from scratch
        trainable_base: Whether to make base model trainable (only relevant if weights='imagenet')
    
    Returns:
        Keras Model
    """
    if model_name == 'mobilenet':
        base_model = MobileNetV2(weights=weights, include_top=False, input_shape=input_shape)
    elif model_name == 'resnet50':
        base_model = ResNet50(weights=weights, include_top=False, input_shape=input_shape)
    elif model_name == 'vgg16':
        base_model = VGG16(weights=weights, include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown transfer model: {model_name}")

    # If training from scratch (weights=None), the base should always be trainable
    if weights is None:
        base_model.trainable = True
    else:
        base_model.trainable = trainable_base
    
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=(weights is None)) # Training=True if from scratch
    x = GlobalAveragePooling2D()(x)
    
    # dropout based on model complexity
    if model_name == 'resnet50':
        x = Dropout(0.5)(x)
    else:
        x = Dropout(0.4)(x)
        
    from tensorflow.keras.regularizers import l2
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)
    
    model = Model(inputs, outputs, name=model_name)
    return model


def create_custom_cnn(input_shape=(224, 224, 3), num_classes=3):
    """
    Create a simple custom CNN from scratch
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name='custom_cnn')
    
    return model


def get_model(model_name, input_shape=(224, 224, 3), num_classes=3, mode='transfer'):
    """
    Factory function to get model by name and mode
    
    Args:
        model_name: Name of the model
        input_shape: Input image shape
        num_classes: Number of output classes
        mode: 'transfer' (use ImageNet weights) or 'scratch' (random initialization)
    
    Returns:
        Keras Model
    """
    model_name = model_name.lower()
    
    if model_name == 'custom_cnn':
        return create_custom_cnn(input_shape, num_classes)
    
    weights = 'imagenet' if mode == 'transfer' else None
    trainable_base = False # Start frozen for transfer learning
    
    return create_transfer_model(
        model_name, 
        input_shape, 
        num_classes, 
        weights=weights, 
        trainable_base=trainable_base
    )


def get_fine_tune_layer(model_name):
    """
    Get the layer index from which to start fine-tuning
    """
    fine_tune_at = {
        'mobilenet': 100,
        'resnet50': 140,
        'vgg16': 15,
        'custom_cnn': 0
    }
    return fine_tune_at.get(model_name.lower(), 0)
