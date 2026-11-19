"""
CropGuard AI - U-Net Segmentation Training & Loss Functions
Implements custom loss functions and callbacks for foliar lesion segmentation:
- Binary Cross-Entropy (BCE) Loss
- Dice Loss (Sørensen-Dice loss combating severe foliar lesion foreground/background imbalance)
- Combined BCE + Dice Loss
- Training callback factories (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger)
"""

from pathlib import Path
from typing import Optional, List, Union
import tensorflow as tf
import keras
from keras import ops


@keras.saving.register_keras_serializable(package="cropguard.segmentation")
def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Computes Sørensen-Dice loss between ground-truth binary masks and predicted probabilities:
    Dice = (2 * sum(y_true * y_pred) + smooth) / (sum(y_true) + sum(y_pred) + smooth)
    Loss = 1.0 - Dice
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Sum over spatial and channel axes [1, 2, 3], preserving batch dimension [0]
    axes = list(range(1, tf.rank(y_true)))
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(1.0 - dice)


@keras.saving.register_keras_serializable(package="cropguard.segmentation")
def bce_dice_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    smooth: float = 1e-6
) -> tf.Tensor:
    """
    Combined BCE + Dice Loss:
    Balances pixel-wise classification cross-entropy with global structural overlap (Dice).
    Particularly resilient to extreme class imbalance where foliar lesions cover 1-10% of leaf area.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # 1. Pixel-wise binary cross-entropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)

    # 2. Dice loss
    d_loss = dice_loss(y_true, y_pred, smooth=smooth)

    return bce_weight * bce + dice_weight * d_loss


@keras.saving.register_keras_serializable(package="cropguard.segmentation")
class BCEDiceLoss(keras.losses.Loss):
    """
    Keras Loss class wrapper for Combined BCE + Dice Loss.
    """
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
        name: str = "bce_dice_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return bce_dice_loss(
            y_true, y_pred,
            bce_weight=self.bce_weight,
            dice_weight=self.dice_weight,
            smooth=self.smooth
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
            "smooth": self.smooth,
        })
        return config


def get_segmentation_callbacks(
    checkpoint_path: Union[str, Path],
    csv_log_path: Optional[Union[str, Path]] = None,
    monitor: str = "val_dice_coefficient",
    mode: str = "max",
    patience_early: int = 10,
    patience_lr: int = 5,
    min_lr: float = 1e-6,
    verbose: int = 1
) -> List[keras.callbacks.Callback]:
    """
    Generates standard callbacks for U-Net lesion segmentation:
    - ModelCheckpoint: Saves best checkpoint based on validation Dice score.
    - EarlyStopping: Halts training when validation Dice stops improving.
    - ReduceLROnPlateau: Decays learning rate on plateau.
    - CSVLogger: Records per-epoch training metrics to disk.
    """
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks: List[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=verbose
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience_early,
            restore_best_weights=True,
            verbose=verbose
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=0.5,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=verbose
        )
    ]

    if csv_log_path is not None:
        csv_p = Path(csv_log_path)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(keras.callbacks.CSVLogger(str(csv_p), append=True))

    return callbacks
