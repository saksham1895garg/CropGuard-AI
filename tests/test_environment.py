"""
Environment verification tests for CropGuard AI.

Verifies Python environment, TensorFlow installation, GPU availability,
and actual GPU tensor execution.
"""

import sys
import tensorflow as tf


def test_python_version():
    """Verify that Python version meets project requirements."""
    version_info = sys.version_info
    assert version_info.major == 3, f"Expected Python 3, got Python {version_info.major}"
    assert version_info.minor >= 11, f"Expected Python >= 3.11, got 3.{version_info.minor}"


def test_tensorflow_installed():
    """Verify that TensorFlow is imported with CUDA build enabled."""
    assert tf.__version__.startswith("2.22"), f"Unexpected TF version: {tf.__version__}"
    build_info = tf.sysconfig.get_build_info()
    assert build_info.get("is_cuda_build", False) is True, "TensorFlow build lacks CUDA support"


def test_gpu_device_available():
    """Verify that at least one physical GPU is registered with TensorFlow."""
    gpus = tf.config.list_physical_devices("GPU")
    assert len(gpus) > 0, "No physical GPU device was detected by TensorFlow"


def test_gpu_tensor_computation_executes_on_gpu():
    """
    Verify that an actual tensor operation is computed on the GPU.

    This test confirms that TensorFlow not only lists the device, but actively
    allocates device memory and places the operation output on /device:GPU:0.
    """
    gpus = tf.config.list_physical_devices("GPU")
    assert len(gpus) > 0, "Cannot test GPU computation without an available GPU"

    # Configure memory growth if not already initialized
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass  # Physical devices can only be configured once before tensor allocation

    # Perform matrix multiplication on GPU:0
    with tf.device("/GPU:0"):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)
        c = tf.matmul(a, b)

    # Validate output tensor properties
    assert c.shape == (2, 2), f"Unexpected result shape: {c.shape}"
    assert "GPU" in c.device.upper(), f"Tensor was not placed on GPU device, got: {c.device}"

    # Verify computation numerical correctness
    # [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]] = [[19, 22], [43, 50]]
    expected = [[19.0, 22.0], [43.0, 50.0]]
    assert c.numpy().tolist() == expected, f"Numerical mismatch: {c.numpy().tolist()} != {expected}"
