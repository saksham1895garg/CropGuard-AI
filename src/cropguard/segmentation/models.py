"""
CropGuard AI - U-Net Foliar Lesion Segmentation Architecture
Implements the standard U-Net architecture (Ronneberger et al., 2015) for
pixel-level foliar pathology segmentation from leaf imagery.

Architecture Overview:
- Contracting Encoder: Successive Conv2D-BN-ReLU-Conv2D-BN-ReLU blocks + MaxPooling2D,
  extracting high-level contextual semantics while halving spatial resolution at each stage.
- Bottleneck: Deepest feature representation capturing global leaf context.
- Expanding Decoder: Conv2DTranspose (or UpSampling2D) blocks doubling spatial resolution,
  concatenating encoder skip connections to preserve high-resolution spatial details
  (lesion boundaries, necrotic margins, punctate lesions).
- Segmentation Head: 1x1 Conv2D with Sigmoid activation for binary lesion detection
  (output shape: [batch, H, W, 1]).
"""

from typing import Tuple, List
import tensorflow as tf
import keras
from keras import layers


def _conv_block(x: tf.Tensor, filters: int, kernel_size: int = 3, use_batchnorm: bool = True, name: str = "conv") -> tf.Tensor:
    """
    Standard double-convolution block:
    [Conv2D -> (BatchNorm) -> ReLU] x 2 with 'same' padding and He normal initialization.
    """
    x = layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name}_conv1"
    )(x)
    if use_batchnorm:
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name}_conv2"
    )(x)
    if use_batchnorm:
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)

    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 1,
    base_filters: int = 32,
    depth: int = 4,
    use_batchnorm: bool = True,
    use_transpose: bool = True,
    dropout_rate: float = 0.0,
    model_name: str = "unet_foliar_lesion"
) -> keras.Model:
    """
    Builds a standard Keras Functional U-Net model for foliar lesion segmentation.

    Args:
        input_shape: Shape of input RGB leaf images (H, W, C), default (256, 256, 3).
        num_classes: Number of segmentation mask output channels (1 for binary lesion segmentation).
        base_filters: Base filter count for the first encoder stage (doubles each stage).
        depth: Number of encoder/decoder stages (default 4 -> filters 32, 64, 128, 256; bottleneck 512).
        use_batchnorm: Whether to apply Batch Normalization after convolutions.
        use_transpose: If True, uses Conv2DTranspose for learnable upsampling; if False, UpSampling2D.
        dropout_rate: Optional spatial dropout rate for bottleneck/decoder regularization.
        model_name: Name of the resulting Keras Model.

    Returns:
        Compiled or uncompiled keras.Model with input_shape and output shape (H, W, num_classes).
    """
    inputs = keras.Input(shape=input_shape, name="leaf_image_input")

    # Encoder (Contracting Path)
    skip_connections: List[tf.Tensor] = []
    x = inputs

    filter_counts = [base_filters * (2 ** i) for i in range(depth)]

    for i, filters in enumerate(filter_counts):
        x = _conv_block(x, filters=filters, use_batchnorm=use_batchnorm, name=f"enc_stage{i+1}")
        skip_connections.append(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"enc_pool{i+1}")(x)

    # Bottleneck
    bottleneck_filters = base_filters * (2 ** depth)
    x = _conv_block(x, filters=bottleneck_filters, use_batchnorm=use_batchnorm, name="bottleneck")
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="bottleneck_dropout")(x)

    # Decoder (Expanding Path)
    for i in reversed(range(depth)):
        filters = filter_counts[i]
        stage_idx = depth - i

        if use_transpose:
            x = layers.Conv2DTranspose(
                filters,
                kernel_size=(2, 2),
                strides=(2, 2),
                padding="same",
                kernel_initializer="he_normal",
                name=f"dec_up{stage_idx}"
            )(x)
        else:
            x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear", name=f"dec_up{stage_idx}")(x)
            x = layers.Conv2D(filters, kernel_size=(1, 1), padding="same", name=f"dec_up_proj{stage_idx}")(x)

        skip = skip_connections[i]
        x = layers.Concatenate(axis=-1, name=f"dec_concat{stage_idx}")([x, skip])
        x = _conv_block(x, filters=filters, use_batchnorm=use_batchnorm, name=f"dec_stage{stage_idx}")
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate, name=f"dec_dropout{stage_idx}")(x)

    # Output Segmentation Head
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        padding="same",
        activation=activation,
        dtype="float32",
        name="lesion_mask_output"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model
