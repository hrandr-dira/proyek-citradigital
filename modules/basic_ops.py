"""
modules/basic_ops.py
Operasi dasar citra: grayscale, biner, aritmatika, logika
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Mengubah gambar BGR/RGB menjadi grayscale."""
    if len(image.shape) == 2:
        return image  # sudah grayscale
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_binary(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Mengubah gambar menjadi citra biner menggunakan thresholding.
    Threshold: 0–255 (default 127)
    """
    gray = to_grayscale(image)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def arithmetic_operation(
    img1: np.ndarray,
    img2: np.ndarray,
    operation: str,
    scalar: float = 50.0,
) -> np.ndarray:
    """
    Operasi aritmatika pada citra.
    operation: 'add', 'subtract', 'multiply', 'divide',
               'add_scalar', 'subtract_scalar'
    """
    # Pastikan kedua gambar berukuran sama jika diperlukan
    if operation in ("add", "subtract", "multiply", "divide"):
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        a = img1[:h, :w].astype(np.float32)
        b = img2[:h, :w].astype(np.float32)

        if operation == "add":
            result = cv2.add(img1[:h, :w], img2[:h, :w])
        elif operation == "subtract":
            result = cv2.subtract(img1[:h, :w], img2[:h, :w])
        elif operation == "multiply":
            result = np.clip(a * b / 255.0, 0, 255).astype(np.uint8)
        elif operation == "divide":
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.where(b != 0, np.clip(a / b * 128, 0, 255), 0).astype(
                    np.uint8
                )
    elif operation == "add_scalar":
        result = cv2.add(img1, np.full_like(img1, int(scalar)))
    elif operation == "subtract_scalar":
        result = cv2.subtract(img1, np.full_like(img1, int(scalar)))
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return result


def logical_operation(
    img1: np.ndarray,
    img2: np.ndarray,
    operation: str,
) -> np.ndarray:
    """
    Operasi logika pada citra.
    operation: 'AND', 'OR', 'XOR', 'NOT'
    """
    if operation == "NOT":
        return cv2.bitwise_not(img1)

    # Resize img2 agar sama dengan img1
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if operation == "AND":
        return cv2.bitwise_and(img1, img2_resized)
    elif operation == "OR":
        return cv2.bitwise_or(img1, img2_resized)
    elif operation == "XOR":
        return cv2.bitwise_xor(img1, img2_resized)
    else:
        raise ValueError(f"Unknown operation: {operation}")
