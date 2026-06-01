"""
modules/morphology.py
Operasi morfologi: dilasi & erosi dengan berbagai Structuring Element (SE)
"""

import cv2
import numpy as np


STRUCTURING_ELEMENTS = {
    "Persegi (3×3)": cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    "Persegi (5×5)": cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    "Persegi (7×7)": cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    "Ellipse (3×3)": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    "Ellipse (5×5)": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    "Ellipse (7×7)": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    "Silang / Cross (3×3)": cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
    "Silang / Cross (5×5)": cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
}


def apply_morphology(
    image: np.ndarray,
    operation: str,
    se_name: str,
    iterations: int = 1,
) -> np.ndarray:
    """
    Terapkan operasi morfologi.
    operation : 'Dilasi' | 'Erosi' | 'Opening' | 'Closing'
    se_name   : nama SE dari STRUCTURING_ELEMENTS
    iterations: jumlah iterasi (1–5)
    """
    if se_name not in STRUCTURING_ELEMENTS:
        raise ValueError(f"Structuring element '{se_name}' tidak ditemukan.")

    kernel = STRUCTURING_ELEMENTS[se_name]

    # Konversi ke grayscale jika diperlukan untuk morfologi
    work_img = image.copy()

    morph_map = {
        "Dilasi": cv2.MORPH_DILATE,
        "Erosi": cv2.MORPH_ERODE,
        "Opening (Erosi → Dilasi)": cv2.MORPH_OPEN,
        "Closing (Dilasi → Erosi)": cv2.MORPH_CLOSE,
    }

    if operation not in morph_map:
        raise ValueError(f"Operasi '{operation}' tidak dikenal.")

    result = cv2.morphologyEx(
        work_img,
        morph_map[operation],
        kernel,
        iterations=iterations,
    )
    return result


def get_se_display(se_name: str) -> np.ndarray:
    """Kembalikan matriks SE untuk ditampilkan."""
    return STRUCTURING_ELEMENTS.get(se_name, np.zeros((3, 3), dtype=np.uint8))


def list_se() -> list:
    return list(STRUCTURING_ELEMENTS.keys())


def list_operations() -> list:
    return ["Dilasi", "Erosi", "Opening (Erosi → Dilasi)", "Closing (Dilasi → Erosi)"]
