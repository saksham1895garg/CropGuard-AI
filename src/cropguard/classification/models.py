"""
CropGuard AI - Foliar Pathology Classification Models.
Implements:
1. Custom 4-stage Foliar CNN (~350k parameters)
2. MobileNetV2 with Depthwise Separable Convolutions & Inverted Residuals (~2.3M params)
3. ResNet50 with Deep Residual Bottleneck blocks (~24M params)
"""

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_custom_foliar_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 38,
    dropout_rate: float = 0.3,
    name: str = "custom_foliar_cnn"
) -> tf.keras.Model:
    """
    Constructs a lightweight, 4-stage convolutional neural network designed specifically
    for foliar disease diagnosis. Employs stacked Conv2D-BatchNorm-ReLU-MaxPool blocks
    followed by GlobalAveragePooling to prevent parameter explosion.

    Architecture:
        - Stage 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) [112x112x32]
        - Stage 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) [56x56x64]
        - Stage 3: Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) [28x28x128]
        - Stage 4: Conv2D(256, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) [14x14x256]
        - Head: GlobalAveragePooling2D -> BatchNorm -> Dense(128, ReLU) -> Dropout -> Dense(38, Softmax)

    Args:
        input_shape: Input spatial dimensions (H, W, C).
        num_classes: Number of foliar pathology classes (38).
        dropout_rate: Dropout probability in classification head.
        name: Name of the model.

    Returns:
        tf.keras.Model ready for compilation.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Stage 1: Low-level edge & gradient filters
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    # Stage 2: Intermediate foliar lesion texture patterns
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    # Stage 3: High-level symptom morphology
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)

    # Stage 4: Complex multi-pathological features
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return models.Model(inputs=inputs, outputs=outputs, name=name)


def build_mobilenet_v2_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 38,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    fine_tune_layers: int = 0,
    dropout_rate: float = 0.3,
    name: str = "mobilenet_v2_cropguard"
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Constructs a transfer learning classifier using MobileNetV2 as the feature extractor.
    MobileNetV2 leverages depthwise separable convolutions and inverted residual bottlenecks,
    providing high computational efficiency suitable for edge deployment.

    Args:
        input_shape: (H, W, C) input tensor shape.
        num_classes: Number of disease categories (38).
        pretrained: If True, initializes backbone with ImageNet weights.
        freeze_backbone: If True, freezes backbone layers during Phase 1 feature extraction.
        fine_tune_layers: Number of top backbone layers to unfreeze for fine-tuning.
        dropout_rate: Dropout rate in classification head.
        name: Name of the resulting model.

    Returns:
        Tuple of (full_model, backbone).
    """
    weights = "imagenet" if pretrained else None
    from tensorflow.keras.applications import MobileNetV2

    backbone = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )
    backbone.name = "mobilenetv2_backbone"

    if freeze_backbone:
        backbone.trainable = False
        if fine_tune_layers > 0:
            # Unfreeze top N layers
            backbone.trainable = True
            for layer in backbone.layers[:-fine_tune_layers]:
                layer.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    features = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(features)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    full_model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return full_model, backbone


def build_resnet50_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 38,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    fine_tune_layers: int = 0,
    dropout_rate: float = 0.4,
    name: str = "resnet50_cropguard"
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Constructs a transfer learning classifier using ResNet50 as the backbone.
    ResNet50 uses residual identity shortcut connections [y = F(x) + x] to overcome
    the vanishing gradient problem across 50 deep convolutional layers.

    Args:
        input_shape: (H, W, C) input tensor shape.
        num_classes: Number of disease categories (38).
        pretrained: If True, initializes backbone with ImageNet weights.
        freeze_backbone: If True, freezes backbone layers during Phase 1 feature extraction.
        fine_tune_layers: Number of top backbone layers to unfreeze for fine-tuning.
        dropout_rate: Dropout rate in classification head.
        name: Name of the resulting model.

    Returns:
        Tuple of (full_model, backbone).
    """
    weights = "imagenet" if pretrained else None
    from tensorflow.keras.applications import ResNet50

    backbone = ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )
    backbone.name = "resnet50_backbone"

    if freeze_backbone:
        backbone.trainable = False
        if fine_tune_layers > 0:
            backbone.trainable = True
            for layer in backbone.layers[:-fine_tune_layers]:
                layer.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    features = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(features)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    full_model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return full_model, backbone


def set_backbone_trainable_layers(backbone: tf.keras.Model, num_layers_to_unfreeze: int) -> int:
    """
    Fine-tuning helper: unfreezes the top N layers of a pretrained backbone
    while keeping the lower feature extraction layers frozen.

    Args:
        backbone: The base convolutional model.
        num_layers_to_unfreeze: Number of top layers from the end to make trainable.

    Returns:
        Total number of trainable weights in the backbone.
    """
    backbone.trainable = True
    if num_layers_to_unfreeze <= 0:
        backbone.trainable = False
        return 0

    for layer in backbone.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    for layer in backbone.layers[-num_layers_to_unfreeze:]:
        # BatchNormalization layers are typically kept frozen in fine-tuning
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    trainable_weights_count = sum(p.numpy().size for p in backbone.trainable_weights)
    return trainable_weights_count
