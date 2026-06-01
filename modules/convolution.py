"""
modules/convolution.py
Operasi konvolusi dengan berbagai jenis filter
"""

import cv2
import numpy as np


KERNELS = {
    "Sharpening (Standard)": np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]], dtype=np.float32
    ),
    "Sharpening (Strong)": np.array(
        [[-1, -1, -1],
         [-1,  9, -1],
         [-1, -1, -1]], dtype=np.float32
    ),
    "Blurring (Mean 3×3)": np.ones((3, 3), dtype=np.float32) / 9.0,
    "Blurring (Mean 5×5)": np.ones((5, 5), dtype=np.float32) / 25.0,
    "Edge Detection (Laplacian)": np.array(
        [[0,  1, 0],
         [1, -4, 1],
         [0,  1, 0]], dtype=np.float32
    ),
    "Edge Detection (Sobel X)": np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]], dtype=np.float32
    ),
    "Edge Detection (Sobel Y)": np.array(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]], dtype=np.float32
    ),
    "Emboss": np.array(
        [[-2, -1, 0],
         [-1,  1, 1],
         [ 0,  1, 2]], dtype=np.float32
    ),
}


def apply_filter(image: np.ndarray, filter_name: str) -> np.ndarray:
    """
    Terapkan konvolusi dengan kernel yang dipilih.
    Mendukung gambar grayscale maupun RGB.
    """
    if filter_name not in KERNELS:
        raise ValueError(f"Filter '{filter_name}' tidak ditemukan.")

    kernel = KERNELS[filter_name]

    # Gaussian Blur diproses terpisah
    result = cv2.filter2D(image, -1, kernel)

    # Untuk edge detection, kita clip agar tidak negatif
    if "Edge" in filter_name:
        result = np.clip(np.abs(result.astype(np.int32)), 0, 255).astype(np.uint8)

    return result


def get_kernel_display(filter_name: str) -> np.ndarray:
    """Kembalikan kernel sebagai numpy array untuk ditampilkan."""
    return KERNELS.get(filter_name, np.array([[0]]))


def list_filters() -> list:
    return list(KERNELS.keys())
