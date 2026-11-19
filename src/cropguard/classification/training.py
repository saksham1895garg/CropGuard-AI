"""
CropGuard AI - Classification Training Utilities.
Provides standard model compilation, optimizer scheduling, callbacks,
and structured training workflows.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def compile_classifier(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.0
) -> tf.keras.Model:
    """
    Compiles a classification model with Adam optimizer and Sparse Categorical Crossentropy.

    Args:
        model: tf.keras.Model to compile.
        learning_rate: Initial learning rate for Adam.
        label_smoothing: Optional label smoothing epsilon.

    Returns:
        Compiled model.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        label_smoothing=label_smoothing
    )
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )
    return model


def get_training_callbacks(
    checkpoint_path: Optional[Union[str, Path]] = None,
    log_file: Optional[Union[str, Path]] = None,
    tensorboard_dir: Optional[Union[str, Path]] = None,
    patience: int = 5,
    min_lr: float = 1e-6
) -> List[tf.keras.callbacks.Callback]:
    """
    Configures robust training callbacks:
    - EarlyStopping: Halts training when validation loss stops improving and restores best weights.
    - ReduceLROnPlateau: Decays learning rate dynamically upon validation loss plateaus.
    - ModelCheckpoint: Saves the best model weights on disk.
    - CSVLogger: Records per-epoch training metrics to disk.
    - TensorBoard: Saves profiling and loss scalar event logs for visual inspection.

    Args:
        checkpoint_path: Optional file path for best model checkpoint (.keras or .weights.h5).
        log_file: Optional CSV path for epoch logging.
        tensorboard_dir: Optional directory path for TensorBoard event logs.
        patience: Epochs of non-improvement before early stopping.
        min_lr: Lower bound on decayed learning rate.

    Returns:
        List of configured Keras Callback instances.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=max(2, patience // 2),
            min_lr=min_lr,
            verbose=1
        )
    ]

    if checkpoint_path is not None:
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.is_absolute():
            ckpt_p = PROJECT_ROOT / ckpt_p
        ckpt_p.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_p),
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
        )

    if log_file is not None:
        log_p = Path(log_file)
        if not log_p.is_absolute():
            log_p = PROJECT_ROOT / log_p
        log_p.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.CSVLogger(
                filename=str(log_p),
                separator=",",
                append=False
            )
        )

    if tensorboard_dir is not None:
        tb_p = Path(tensorboard_dir)
        if not tb_p.is_absolute():
            tb_p = PROJECT_ROOT / tb_p
        tb_p.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=str(tb_p),
                histogram_freq=0,
                write_graph=True
            )
        )

    return callbacks
