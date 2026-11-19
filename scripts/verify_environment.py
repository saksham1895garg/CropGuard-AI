#!/usr/bin/env python3
"""
CropGuard AI - Environment and GPU Verification Script

This script verifies:
1. Python version and executable path.
2. TensorFlow version and build configuration.
3. NVIDIA GPU detection and physical device listing.
4. GPU memory growth setup.
5. Actual execution of tensor operations on the GPU.
"""

import sys
import tensorflow as tf


def main() -> int:
    print("=" * 60)
    print("CropGuard AI - Environment Verification")
    print("=" * 60)

    # 1. Python Details
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Python Executable : {sys.executable}")

    # 2. TensorFlow Details
    print(f"TensorFlow Version: {tf.__version__}")
    build_info = tf.sysconfig.get_build_info()
    print(f"CUDA Build Enabled: {build_info.get('is_cuda_build', False)}")

    # 3. GPU Detection
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs Detected     : {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  - Device {i}: {gpu.name}")

    if not gpus:
        print("\n[ERROR] No physical GPU detected by TensorFlow.")
        print("Please check NVIDIA drivers and WSL2 GPU pass-through.")
        return 1

    # 4. Enable GPU memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory Growth     : Enabled successfully")
    except RuntimeError as e:
        print(f"Memory Growth     : Warning: {e}")

    # 5. Execute computation on GPU
    print("\nExecuting test computation on /GPU:0...")
    try:
        with tf.device("/GPU:0"):
            matrix_a = tf.random.normal((1000, 1000))
            matrix_b = tf.random.normal((1000, 1000))
            result = tf.matmul(matrix_a, matrix_b)

        # Force execution and inspect tensor
        device_used = result.device
        print(f"Matrix Multiply   : OK (Shape: {result.shape})")
        print(f"Execution Device  : {device_used}")

        if "GPU" not in device_used.upper():
            print(f"\n[ERROR] Operation was scheduled on GPU but resolved on: {device_used}")
            return 1

        print("\n[SUCCESS] Environment verified. TensorFlow GPU acceleration is working!")
        return 0

    except Exception as exc:
        print(f"\n[ERROR] GPU tensor execution failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
