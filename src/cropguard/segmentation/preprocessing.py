"""
CropGuard AI - Foliar Lesion Segmentation Preprocessing Module
Provides reproducible image and mask normalization, binarization,
and spatial alignment for the PlantSeg dataset and U-Net pipeline.
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


def normalize_image(image: Union[np.ndarray, tf.Tensor]) -> Union[np.ndarray, tf.Tensor]:
    """
    Scales RGB leaf imagery to [0.0, 1.0] float32.
    """
    if isinstance(image, np.ndarray):
        img_f = image.astype(np.float32)
        if img_f.max() > 1.0:
            img_f = img_f / 255.0
        return np.clip(img_f, 0.0, 1.0)
    else:
        img_f = tf.cast(image, tf.float32)
        img_max = tf.reduce_max(img_f)
        img_f = tf.cond(img_max > 1.0, lambda: img_f / 255.0, lambda: img_f)
        return tf.clip_by_value(img_f, 0.0, 1.0)


def binarize_mask(mask: Union[np.ndarray, tf.Tensor]) -> Union[np.ndarray, tf.Tensor]:
    """
    Converts raw PlantSeg mask annotations (where background is 0 and lesion pixels
    are marked with integer disease IDs [1, 115]) into binary masks {0.0, 1.0}.
    Ensures trailing channel dimension (H, W, 1).
    """
    if isinstance(mask, np.ndarray):
        # Ensure 2D or 3D
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)
        elif mask.ndim == 3 and mask.shape[-1] > 1:
            # If multi-channel mask (RGB/RGBA), take max or channel 0
            mask = np.expand_dims(mask[..., 0], axis=-1)

        bin_mask = (mask > 0).astype(np.float32)
        return bin_mask
    else:
        mask_t = tf.cast(mask, tf.float32)
        if tf.rank(mask_t) == 2:
            mask_t = tf.expand_dims(mask_t, axis=-1)
        elif tf.rank(mask_t) == 3 and tf.shape(mask_t)[-1] > 1:
            mask_t = tf.expand_dims(mask_t[..., 0], axis=-1)

        bin_mask = tf.cast(mask_t > 0.0, tf.float32)
        return bin_mask


def preprocess_image_and_mask(
    image: Union[np.ndarray, Image.Image],
    mask: Union[np.ndarray, Image.Image],
    target_shape: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard dual preprocessing pipeline for an (image, mask) pair:
    1. Resolves camera EXIF orientation flags.
    2. Aligns spatial dimensions if transposed.
    3. Resizes image using bilinear interpolation to target_shape.
    4. Resizes mask using nearest-neighbor interpolation to target_shape (preserving sharp boundaries).
    5. Normalizes image to [0.0, 1.0] and mask to binary {0.0, 1.0}.
    6. Ensures shape (H, W, 3) for image and (H, W, 1) for mask.

    Returns:
        (image_tensor, mask_tensor) as float32 NumPy arrays.
    """
    # 1. Convert to PIL if necessary
    if isinstance(image, np.ndarray):
        im_pil = Image.fromarray(image.astype(np.uint8))
    else:
        im_pil = image.convert("RGB")

    if isinstance(mask, np.ndarray):
        # squeeze 1-channel if present
        m_arr = mask.squeeze()
        ma_pil = Image.fromarray(m_arr.astype(np.uint8))
    else:
        ma_pil = mask

    # 2. Resolve EXIF orientation for JPEG
    im_pil = ImageOps.exif_transpose(im_pil)

    # 3. Handle dimension swap if mask is transposed relative to image
    iw, ih = im_pil.size
    mw, mh = ma_pil.size
    if (iw, ih) == (mh, mw) and (iw != mw):
        ma_pil = ma_pil.transpose(Image.Transpose.TRANSPOSE)

    # 4. Resize: Image -> Bilinear, Mask -> Nearest
    im_resized = im_pil.resize((target_shape[1], target_shape[0]), Image.Resampling.BILINEAR)
    ma_resized = ma_pil.resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST)

    # 5. Convert to float32 NumPy arrays
    im_arr = np.array(im_resized, dtype=np.float32) / 255.0
    im_arr = np.clip(im_arr, 0.0, 1.0)

    ma_arr = np.array(ma_resized)
    if ma_arr.ndim == 2:
        ma_arr = np.expand_dims(ma_arr, axis=-1)
    elif ma_arr.ndim == 3 and ma_arr.shape[-1] > 1:
        ma_arr = np.expand_dims(ma_arr[..., 0], axis=-1)

    bin_mask = (ma_arr > 0).astype(np.float32)

    return im_arr, bin_mask


def preprocess_image(
    image: Union[np.ndarray, Image.Image],
    target_shape: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Inference-time single image preprocessor.
    """
    if isinstance(image, np.ndarray):
        im_pil = Image.fromarray(image.astype(np.uint8))
    else:
        im_pil = image.convert("RGB")

    im_pil = ImageOps.exif_transpose(im_pil)
    im_resized = im_pil.resize((target_shape[1], target_shape[0]), Image.Resampling.BILINEAR)
    im_arr = np.array(im_resized, dtype=np.float32) / 255.0
    return np.clip(im_arr, 0.0, 1.0)
